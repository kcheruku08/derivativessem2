#!/usr/bin/env python3
"""
Dispersion Trading Backtester
==============================
Strategy: Sell single-stock vol on AI semiconductors (NVDA, AVGO, TSM)
          Buy index vol on SMH.
This is a short-correlation trade that profits when the stocks move
independently and loses when they crash together.

Signal: Implied correlation derived from index vs constituent variances.
When implied correlation is high (>75th pctl) -> enter dispersion trade.
When it normalises (below median) -> exit.
"""

import os
import sys
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# yfinance cache setup — isolated dir to avoid lock conflicts
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yfinance_cache_disp")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    import yfinance as yf
    yf.set_tz_cache_location(CACHE_DIR)
except Exception:
    import yfinance as yf

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CONSTITUENTS = ["NVDA", "AVGO", "TSM"]
INDEX = "SMH"
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])

VOL_WINDOW = 21          # rolling window for realised vol (business days)
IV_PREMIUM = 1.10        # scale realised vol by 1.1 as IV proxy
LOOKBACK_PCTL = 252      # 1-year rolling window for percentile ranks

ENTRY_PCTL = 75          # enter when implied corr > 75th percentile
EXIT_PCTL = 50           # exit  when implied corr < median

MIN_HOLD = 5             # minimum hold in trading days
MAX_HOLD = 21            # maximum hold (force exit)

SLIPPAGE_PCT = 0.015     # 1.5 % of straddle premium on entry & exit
THETA_SOLD = 0.0004      # +0.04 % of notional per day (collecting)
THETA_BOUGHT = 0.00035   # -0.035 % of notional per day (paying)
STRADDLE_HOLD_DAYS = 21  # assumed DTE for straddle premium estimate

INDEX_NOTIONAL = 1_000_000  # $1M notional on index leg

DOWNLOAD_YEARS = 5

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data() -> pd.DataFrame:
    """Download daily close prices for constituents + index, one ticker at a time."""
    end = dt.date.today()
    start = end - dt.timedelta(days=DOWNLOAD_YEARS * 365 + 30)

    tickers = CONSTITUENTS + [INDEX]
    frames = {}
    for tkr in tickers:
        print(f"  Downloading {tkr} …")
        df = yf.download(tkr, start=str(start), end=str(end),
                         progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data returned for {tkr}")
        # Handle multi-level columns from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[tkr] = df["Close"].rename(tkr)

    prices = pd.concat(frames.values(), axis=1).dropna()
    prices.index = pd.to_datetime(prices.index)
    print(f"  Price matrix: {prices.shape[0]} rows  x  {prices.shape[1]} cols")
    return prices


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def compute_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily implied-correlation signal from rolling realised vols.

    implied_corr = (sigma_idx^2 - sum(w_i^2 * sigma_i^2))
                   / (2 * sum_{i<j} w_i * w_j * sigma_i * sigma_j)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()

    # Annualised rolling realised vol, scaled by IV premium
    rv = log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252) * IV_PREMIUM

    sig = pd.DataFrame(index=rv.index)
    sig["index_var"] = rv[INDEX] ** 2

    for i, c in enumerate(CONSTITUENTS):
        sig[f"var_{c}"] = rv[c] ** 2
        sig[f"vol_{c}"] = rv[c]

    sig["index_vol"] = rv[INDEX]

    # Weighted sum of constituent variances
    sig["sum_wi2_vari"] = sum(
        WEIGHTS[i] ** 2 * sig[f"var_{CONSTITUENTS[i]}"]
        for i in range(len(CONSTITUENTS))
    )

    # Cross terms: 2 * sum_{i<j} w_i w_j sqrt(var_i * var_j)
    cross = pd.Series(0.0, index=sig.index)
    for i in range(len(CONSTITUENTS)):
        for j in range(i + 1, len(CONSTITUENTS)):
            cross += (
                WEIGHTS[i] * WEIGHTS[j]
                * sig[f"vol_{CONSTITUENTS[i]}"]
                * sig[f"vol_{CONSTITUENTS[j]}"]
            )
    sig["cross_term"] = 2.0 * cross

    # Implied correlation
    numerator = sig["index_var"] - sig["sum_wi2_vari"]
    denominator = sig["cross_term"]
    sig["implied_corr"] = numerator / denominator
    # Clip to [-1, 1] for sanity
    sig["implied_corr"] = sig["implied_corr"].clip(-1, 1)

    # Rolling percentile rank
    sig["corr_pctl"] = (
        sig["implied_corr"]
        .rolling(LOOKBACK_PCTL, min_periods=VOL_WINDOW * 2)
        .rank(pct=True) * 100
    )

    # Keep returns for P&L calc
    for c in CONSTITUENTS + [INDEX]:
        sig[f"ret_{c}"] = log_ret[c]

    sig.dropna(inplace=True)
    return sig


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
def run_backtest(sig: pd.DataFrame):
    """
    Walk through dates. When implied corr > 75th pctl, enter dispersion
    trade. Exit when corr < 50th pctl or max hold reached.
    """
    dates = sig.index.tolist()
    n = len(dates)

    # --- Vega-neutral sizing ---------------------------------------------------
    # Index notional is fixed. Scale constituent notional so total
    # vega sold = vega bought.
    # Vega ~ notional * sqrt(T/252) for ATM straddle (approx).
    # Since we sell 3 constituents and buy 1 index, we want:
    #   sum(const_notional_i * vol_i) = index_notional * vol_index
    # For equal-weight, each constituent notional = index_notional * vol_index / (3 * vol_i)
    # We compute this at entry and hold constant for the trade.

    # Trade log
    trades = []
    equity = []
    cum_pnl = 0.0

    # Tracking
    in_trade = False
    entry_idx = None
    hold_days = 0
    trade_theta = 0.0
    trade_gamma = 0.0
    trade_slip = 0.0
    const_notionals = {}
    entry_corr = None

    def _straddle_premium(notional, iv):
        """Approximate ATM straddle premium as notional * IV * sqrt(T/252)."""
        return notional * iv * np.sqrt(STRADDLE_HOLD_DAYS / 252)

    def _slippage(notional, iv):
        """Slippage = SLIPPAGE_PCT * straddle premium (not raw notional)."""
        return SLIPPAGE_PCT * _straddle_premium(notional, iv)

    for t in range(n):
        today = dates[t]
        row = sig.iloc[t]
        daily_pnl = 0.0

        if in_trade:
            hold_days += 1

            # Per-leg tracking for this day
            leg_pnl = {}

            # --- Daily P&L for each constituent leg (short straddle) ----------
            for c in CONSTITUENTS:
                notional = const_notionals[c]
                iv_c = row[f"vol_{c}"]
                daily_ret = row[f"ret_{c}"]

                # Theta collected
                theta = THETA_SOLD * notional
                trade_theta += theta

                # Gamma P&L using user-specified formula:
                #   gamma_pnl = position * gamma * (daily_return^2 - implied_daily_var)
                #   gamma = 0.5 / (IV * sqrt(1/252))
                # "position" = straddle vega notional = notional * IV * sqrt(T/252)
                # For short straddle, sign is negative (we lose when realised > implied)
                iv_daily = iv_c / np.sqrt(252)
                if iv_c > 1e-10:
                    gamma_approx = 0.5 / (iv_c * np.sqrt(1.0 / 252))
                    # Scale position to vega-sized: vega ~ notional * sqrt(1/252)
                    vega_position = notional * np.sqrt(1.0 / 252)
                    gamma_pnl = -vega_position * gamma_approx * (daily_ret ** 2 - iv_daily ** 2)
                else:
                    gamma_pnl = 0.0
                trade_gamma += gamma_pnl
                leg_pnl[c] = theta + gamma_pnl

                daily_pnl += theta + gamma_pnl

            # --- Daily P&L for index leg (long straddle) --------------------
            iv_idx = row["index_vol"]
            ret_idx = row[f"ret_{INDEX}"]

            theta_idx = -THETA_BOUGHT * INDEX_NOTIONAL
            trade_theta += theta_idx

            iv_daily_idx = iv_idx / np.sqrt(252)
            if iv_idx > 1e-10:
                gamma_approx_idx = 0.5 / (iv_idx * np.sqrt(1.0 / 252))
                vega_position_idx = INDEX_NOTIONAL * np.sqrt(1.0 / 252)
                gamma_pnl_idx = vega_position_idx * gamma_approx_idx * (ret_idx ** 2 - iv_daily_idx ** 2)
            else:
                gamma_pnl_idx = 0.0
            trade_gamma += gamma_pnl_idx
            leg_pnl[INDEX] = theta_idx + gamma_pnl_idx

            daily_pnl += theta_idx + gamma_pnl_idx

            # --- Exit conditions -------------------------------------------
            exit_signal = row["corr_pctl"] < EXIT_PCTL and hold_days >= MIN_HOLD
            forced_exit = hold_days >= MAX_HOLD

            if exit_signal or forced_exit:
                # Slippage on exit — applied to straddle premium, not raw notional
                exit_slip = 0.0
                for c in CONSTITUENTS:
                    exit_slip -= _slippage(const_notionals[c], row[f"vol_{c}"])
                exit_slip -= _slippage(INDEX_NOTIONAL, iv_idx)
                trade_slip += exit_slip
                daily_pnl += exit_slip

                total_trade_pnl = trade_theta + trade_gamma + trade_slip
                trades.append({
                    "entry_date": dates[entry_idx],
                    "exit_date": today,
                    "hold_days": hold_days,
                    "impl_corr_entry": entry_corr,
                    "impl_corr_exit": row["implied_corr"],
                    "corr_pctl_entry": sig.iloc[entry_idx]["corr_pctl"],
                    "corr_pctl_exit": row["corr_pctl"],
                    "theta_pnl": trade_theta,
                    "gamma_pnl": trade_gamma,
                    "slippage_cost": trade_slip,
                    "total_pnl": total_trade_pnl,
                    "exit_reason": "forced" if forced_exit and not exit_signal else "signal",
                    "pnl_NVDA": 0.0,
                    "pnl_AVGO": 0.0,
                    "pnl_TSM": 0.0,
                    "pnl_SMH": 0.0,
                })

                in_trade = False
                entry_idx = None

        else:
            # --- Entry conditions -------------------------------------------
            if row["corr_pctl"] > ENTRY_PCTL:
                in_trade = True
                entry_idx = t
                hold_days = 0
                trade_theta = 0.0
                trade_gamma = 0.0
                trade_slip = 0.0
                entry_corr = row["implied_corr"]

                # Vega-neutral sizing
                const_notionals = {}
                for c in CONSTITUENTS:
                    vol_c = row[f"vol_{c}"]
                    vol_idx = row["index_vol"]
                    if vol_c > 1e-10:
                        const_notionals[c] = INDEX_NOTIONAL * vol_idx / (3.0 * vol_c)
                    else:
                        const_notionals[c] = INDEX_NOTIONAL / 3.0

                # Slippage on entry — applied to straddle premium
                entry_slip = 0.0
                for c in CONSTITUENTS:
                    entry_slip -= _slippage(const_notionals[c], row[f"vol_{c}"])
                entry_slip -= _slippage(INDEX_NOTIONAL, row["index_vol"])
                trade_slip += entry_slip
                daily_pnl += entry_slip

        cum_pnl += daily_pnl
        equity.append({
            "date": today,
            "daily_pnl": daily_pnl,
            "cum_pnl": cum_pnl,
            "in_trade": in_trade,
            "implied_corr": row["implied_corr"],
            "corr_pctl": row["corr_pctl"],
        })

    return pd.DataFrame(trades), pd.DataFrame(equity)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def print_results(trades: pd.DataFrame, equity: pd.DataFrame):
    """Print portfolio metrics, attribution, and trade log."""

    print("\n" + "=" * 80)
    print("  DISPERSION TRADING BACKTEST — AI SEMICONDUCTORS")
    print("  Short single-stock vol (NVDA, AVGO, TSM) / Long index vol (SMH)")
    print("=" * 80)

    # --- Portfolio metrics ---------------------------------------------------
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq.set_index("date", inplace=True)

    total_return = eq["cum_pnl"].iloc[-1]
    n_days = len(eq)
    ann_factor = 252 / n_days

    daily_rets = eq["daily_pnl"]
    mean_daily = daily_rets.mean()
    std_daily = daily_rets.std()
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0.0

    # Max drawdown on cum P&L
    running_max = eq["cum_pnl"].cummax()
    drawdown = eq["cum_pnl"] - running_max
    max_dd = drawdown.min()

    n_trades = len(trades)
    if n_trades > 0:
        win_rate = (trades["total_pnl"] > 0).mean() * 100
        avg_pnl = trades["total_pnl"].mean()
        avg_hold = trades["hold_days"].mean()
        best_trade = trades["total_pnl"].max()
        worst_trade = trades["total_pnl"].min()
    else:
        win_rate = avg_pnl = avg_hold = best_trade = worst_trade = 0.0

    pct_time_in_trade = eq["in_trade"].mean() * 100

    print(f"\n{'PORTFOLIO METRICS':^80}")
    print("-" * 80)
    print(f"  Period            : {eq.index[0].date()} to {eq.index[-1].date()}  ({n_days} trading days)")
    print(f"  Total P&L         : ${total_return:>14,.2f}")
    print(f"  Annualised Sharpe : {sharpe:>14.3f}")
    print(f"  Max Drawdown      : ${max_dd:>14,.2f}")
    print(f"  Number of Trades  : {n_trades:>14d}")
    print(f"  Win Rate          : {win_rate:>13.1f}%")
    print(f"  Avg Trade P&L     : ${avg_pnl:>14,.2f}")
    print(f"  Avg Hold (days)   : {avg_hold:>14.1f}")
    print(f"  Best Trade        : ${best_trade:>14,.2f}")
    print(f"  Worst Trade       : ${worst_trade:>14,.2f}")
    print(f"  % Time in Trade   : {pct_time_in_trade:>13.1f}%")
    print(f"  Index Notional    : ${INDEX_NOTIONAL:>14,.0f}")

    # --- P&L Attribution -----------------------------------------------------
    if n_trades > 0:
        total_theta = trades["theta_pnl"].sum()
        total_gamma = trades["gamma_pnl"].sum()
        total_slip = trades["slippage_cost"].sum()
    else:
        total_theta = total_gamma = total_slip = 0.0

    print(f"\n{'P&L ATTRIBUTION':^80}")
    print("-" * 80)
    print(f"  Theta Collected   : ${total_theta:>14,.2f}")
    print(f"  Gamma P&L         : ${total_gamma:>14,.2f}")
    print(f"  Slippage Costs    : ${total_slip:>14,.2f}")
    print(f"  -----------------------------------------")
    print(f"  Net P&L           : ${total_theta + total_gamma + total_slip:>14,.2f}")

    # --- Trade Log -----------------------------------------------------------
    if n_trades > 0:
        print(f"\n{'TRADE LOG':^80}")
        print("-" * 80)
        print(f"  {'#':>3}  {'Entry':>10}  {'Exit':>10}  {'Days':>4}  "
              f"{'Corr In':>7}  {'Corr Out':>8}  {'Theta':>10}  "
              f"{'Gamma':>10}  {'Slip':>10}  {'Total':>11}  {'Reason':>7}")
        print("  " + "-" * 106)
        for i, r in trades.iterrows():
            entry_str = pd.Timestamp(r["entry_date"]).strftime("%Y-%m-%d")
            exit_str = pd.Timestamp(r["exit_date"]).strftime("%Y-%m-%d")
            print(f"  {i + 1:>3}  {entry_str}  {exit_str}  {r['hold_days']:>4.0f}  "
                  f"{r['impl_corr_entry']:>7.3f}  {r['impl_corr_exit']:>8.3f}  "
                  f"${r['theta_pnl']:>9,.0f}  ${r['gamma_pnl']:>9,.0f}  "
                  f"${r['slippage_cost']:>9,.0f}  ${r['total_pnl']:>10,.0f}  "
                  f"{r['exit_reason']:>7}")

    print("\n" + "=" * 80)


def save_outputs(trades: pd.DataFrame, equity: pd.DataFrame, out_dir: str):
    """Write trade_log.csv and equity_curve.csv."""
    os.makedirs(out_dir, exist_ok=True)

    trade_path = os.path.join(out_dir, "trade_log.csv")
    equity_path = os.path.join(out_dir, "equity_curve.csv")

    trades.to_csv(trade_path, index=False)
    equity.to_csv(equity_path, index=False)

    print(f"\n  Saved: {trade_path}")
    print(f"  Saved: {equity_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  DOWNLOADING DATA")
    print("=" * 80)

    prices = download_data()

    print("\n" + "=" * 80)
    print("  COMPUTING SIGNALS")
    print("=" * 80)

    sig = compute_signals(prices)
    print(f"  Signal matrix: {sig.shape[0]} rows")
    print(f"  Implied corr — mean: {sig['implied_corr'].mean():.3f}  "
          f"std: {sig['implied_corr'].std():.3f}  "
          f"min: {sig['implied_corr'].min():.3f}  "
          f"max: {sig['implied_corr'].max():.3f}")

    print("\n" + "=" * 80)
    print("  RUNNING BACKTEST")
    print("=" * 80)

    trades, equity = run_backtest(sig)
    print(f"  Completed. {len(trades)} trades executed.")

    print_results(trades, equity)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_output")
    save_outputs(trades, equity, out_dir)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
