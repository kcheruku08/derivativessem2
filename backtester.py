"""
Event-Driven Backtester for SMH/QQQ Relative Volatility Pairs Strategy
-----------------------------------------------------------------------
Uses free Yahoo Finance data (no API key required).
Produces a detailed trade log with entry/exit dates, Greeks, P&L attribution.
"""

import os
import sys
import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

YF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yfinance_cache_bt")
os.makedirs(YF_CACHE_DIR, exist_ok=True)
try:
    yf.set_tz_cache_location(YF_CACHE_DIR)
except Exception:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class StrategyConfig:
    # Data
    backtest_years: int = 5
    iv_rv_lookback: int = 21

    # Signal
    ols_window: int = 60
    z_ema_halflife: int = 15
    use_asymmetric_zscore: bool = True

    # Entry/Exit
    entry_thresh: float = 1.0
    exit_thresh: float = 0.35
    min_hold_days: int = 21
    reentry_buffer: float = 0.15
    cooldown_days: int = 3

    # Costs & Greeks (per unit notional)
    options_slippage_pct: float = 0.01
    stock_slippage_pct: float = 0.001
    daily_carry_cost: float = 0.0003
    smh_vega: float = 1.25
    qqq_vega: float = 1.00
    smh_theta_daily: float = 0.00022
    qqq_theta_daily: float = 0.00018
    smh_gamma: float = 0.10
    qqq_gamma: float = 0.08


# ============================================================
# DATA LOADING
# ============================================================

def load_market_data(config: StrategyConfig) -> pd.DataFrame:
    """Download SMH, QQQ, VXN from Yahoo Finance and construct IV series."""
    period = f"{config.backtest_years}y"

    # Download separately to avoid cache lock issues
    smh_raw = yf.download("SMH", period=period, interval="1d", auto_adjust=True, progress=False)
    qqq_raw = yf.download("QQQ", period=period, interval="1d", auto_adjust=True, progress=False)
    vxn_raw = yf.download("^VXN", period=period, interval="1d", auto_adjust=True, progress=False)

    if smh_raw.empty or qqq_raw.empty or vxn_raw.empty:
        raise RuntimeError(
            f"Failed to download data. Rows: SMH={len(smh_raw)}, QQQ={len(qqq_raw)}, VXN={len(vxn_raw)}. "
            "Check your internet connection or try again."
        )

    df = pd.DataFrame({
        "SMH": smh_raw["Close"].squeeze(),
        "QQQ": qqq_raw["Close"].squeeze(),
        "VXN": vxn_raw["Close"].squeeze(),
    }).dropna()

    df["smh_ret"] = np.log(df["SMH"]).diff()
    df["qqq_ret"] = np.log(df["QQQ"]).diff()

    rv_s = df["smh_ret"].rolling(config.iv_rv_lookback).std() * np.sqrt(252)
    rv_q = df["qqq_ret"].rolling(config.iv_rv_lookback).std() * np.sqrt(252)

    df["qqq_iv"] = df["VXN"] / 100.0

    qqq_premium = df["qqq_iv"] / rv_q.replace(0.0, np.nan)
    lagged_premium = (
        qqq_premium.replace([np.inf, -np.inf], np.nan)
        .ewm(halflife=63, adjust=False)
        .mean()
        .shift(1)
        .fillna(1.12)
        .clip(0.70, 2.00)
    )
    warmup_proxy = (df["qqq_iv"] * 1.15).clip(0.08, 2.0)
    df["smh_iv"] = (rv_s * lagged_premium).clip(0.08, 2.0).fillna(warmup_proxy).ffill()

    df["smh_ret"] = df["smh_ret"].fillna(0.0)
    df["qqq_ret"] = df["qqq_ret"].fillna(0.0)

    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================

def compute_signal(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Compute residual z-score signal and add to dataframe."""
    smh_iv = df["smh_iv"].values.astype(float)
    qqq_iv = df["qqq_iv"].values.astype(float)
    n = len(smh_iv)
    window = config.ols_window

    raw = np.full(n, np.nan)
    for i in range(window, n):
        x = qqq_iv[i - window:i]
        y = smh_iv[i - window:i]
        slope, intercept, *_ = stats.linregress(x, y)
        raw[i] = smh_iv[i] - (slope * qqq_iv[i] + intercept)

    z_sym = np.full(n, np.nan)
    for i in range(window * 2, n):
        window_resids = raw[i - window:i]
        mu, sig = np.nanmean(window_resids), np.nanstd(window_resids)
        if sig > 0:
            z_sym[i] = (raw[i] - mu) / sig

    if config.use_asymmetric_zscore:
        z_out = np.full(n, np.nan)
        min_side_obs = 5
        for i in range(window * 2, n):
            resid_now = raw[i]
            if np.isnan(resid_now):
                continue
            window_resids = raw[i - window:i]
            if np.all(np.isnan(window_resids)):
                continue
            mu = np.nanmean(window_resids)
            centered = window_resids - mu
            pos = centered[centered > 0]
            neg = centered[centered < 0]
            sig_sym = np.nanstd(centered)
            if resid_now >= mu:
                sig_side = np.nanstd(pos) if len(pos) >= min_side_obs else sig_sym
            else:
                sig_side = np.nanstd(neg) if len(neg) >= min_side_obs else sig_sym
            if sig_side > 0 and np.isfinite(sig_side):
                z_out[i] = (resid_now - mu) / sig_side
        z_out[:window * 2] = z_sym[:window * 2]
    else:
        z_out = z_sym

    smoothed = pd.Series(z_out).interpolate(limit_direction="both").ewm(
        halflife=config.z_ema_halflife, adjust=False
    ).mean().values

    df["raw_residual"] = raw
    df["z_score"] = smoothed
    return df


# ============================================================
# TRADE LOG
# ============================================================

@dataclass
class Trade:
    trade_id: int
    direction: str  # "LONG_SMH_VOL" or "SHORT_SMH_VOL"
    entry_date: datetime.date = None
    exit_date: Optional[datetime.date] = None
    entry_z: float = 0.0
    exit_z: float = 0.0
    hold_days: int = 0

    # IV at entry
    smh_iv_entry: float = 0.0
    qqq_iv_entry: float = 0.0
    smh_iv_exit: float = 0.0
    qqq_iv_exit: float = 0.0

    # P&L attribution (cumulative over hold period)
    pnl_underlying: float = 0.0
    pnl_vega: float = 0.0
    pnl_gamma: float = 0.0
    pnl_theta: float = 0.0
    cost_slippage: float = 0.0
    cost_carry: float = 0.0
    pnl_total: float = 0.0

    # Running accumulators (not exported)
    _daily_pnls: list = field(default_factory=list, repr=False)


# ============================================================
# EVENT-DRIVEN BACKTESTER
# ============================================================

class Backtester:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.trades: list[Trade] = []
        self.daily_returns: list[float] = []
        self.daily_positions: list[int] = []
        self.daily_equity: list[float] = []
        self.dates: list = []

    def run(self, df: pd.DataFrame) -> "Backtester":
        cfg = self.config
        n = len(df)

        z_scores = df["z_score"].values
        smh_ret = df["smh_ret"].values
        qqq_ret = df["qqq_ret"].values
        smh_iv = df["smh_iv"].values
        qqq_iv = df["qqq_iv"].values
        dates = df.index.tolist()

        pos = 0
        hold_days = 0
        cooldown = 0
        trade_id = 0
        current_trade: Optional[Trade] = None
        equity = 1.0

        for i in range(1, n):
            z = z_scores[i - 1]  # signal from prior close
            daily_pnl = 0.0

            if np.isnan(z):
                self.daily_returns.append(0.0)
                self.daily_positions.append(0)
                self.daily_equity.append(equity)
                self.dates.append(dates[i])
                continue

            # --- EXIT LOGIC ---
            exited = False
            if pos == -1 and z < cfg.exit_thresh and hold_days >= cfg.min_hold_days:
                exited = True
            elif pos == 1 and z > -cfg.exit_thresh and hold_days >= cfg.min_hold_days:
                exited = True

            if exited and current_trade is not None:
                current_trade.exit_date = dates[i]
                current_trade.exit_z = z
                current_trade.smh_iv_exit = smh_iv[i]
                current_trade.qqq_iv_exit = qqq_iv[i]
                current_trade.hold_days = hold_days
                current_trade.cost_slippage += cfg.options_slippage_pct
                current_trade.pnl_total = (
                    current_trade.pnl_underlying
                    + current_trade.pnl_vega
                    + current_trade.pnl_gamma
                    + current_trade.pnl_theta
                    - current_trade.cost_slippage
                    - current_trade.cost_carry
                )
                self.trades.append(current_trade)
                daily_pnl -= cfg.options_slippage_pct
                current_trade = None
                pos = 0
                hold_days = 0
                cooldown = cfg.cooldown_days

            # --- ENTRY LOGIC ---
            if pos == 0:
                if cooldown > 0:
                    cooldown -= 1
                else:
                    enter_short = cfg.entry_thresh + cfg.reentry_buffer
                    enter_long = -(cfg.entry_thresh + cfg.reentry_buffer)
                    if z > enter_short:
                        pos = -1
                        hold_days = 1
                        trade_id += 1
                        current_trade = Trade(
                            trade_id=trade_id,
                            direction="SHORT_SMH_VOL",
                            entry_date=dates[i],
                            entry_z=z,
                            smh_iv_entry=smh_iv[i],
                            qqq_iv_entry=qqq_iv[i],
                        )
                        current_trade.cost_slippage += cfg.options_slippage_pct
                        daily_pnl -= cfg.options_slippage_pct
                    elif z < enter_long:
                        pos = 1
                        hold_days = 1
                        trade_id += 1
                        current_trade = Trade(
                            trade_id=trade_id,
                            direction="LONG_SMH_VOL",
                            entry_date=dates[i],
                            entry_z=z,
                            smh_iv_entry=smh_iv[i],
                            qqq_iv_entry=qqq_iv[i],
                        )
                        current_trade.cost_slippage += cfg.options_slippage_pct
                        daily_pnl -= cfg.options_slippage_pct

            # --- DAILY P&L (if in position) ---
            if pos != 0 and current_trade is not None:
                if hold_days > 1 or (hold_days == 1 and not exited):
                    smh_side = pos
                    qqq_side = -pos

                    # Underlying leg
                    und_pnl = 0.5 * pos * (smh_ret[i] - qqq_ret[i])

                    # Vega P&L (IV point change)
                    d_smh_iv = smh_iv[i] - smh_iv[i - 1]
                    d_qqq_iv = qqq_iv[i] - qqq_iv[i - 1]
                    vega_pnl = (smh_side * cfg.smh_vega * d_smh_iv
                                + qqq_side * cfg.qqq_vega * d_qqq_iv)

                    # Gamma P&L
                    gamma_pnl = (smh_side * cfg.smh_gamma * abs(smh_ret[i])
                                 + qqq_side * cfg.qqq_gamma * abs(qqq_ret[i]))

                    # Theta decay
                    theta_pnl = -(abs(smh_side) * cfg.smh_theta_daily
                                  + abs(qqq_side) * cfg.qqq_theta_daily)

                    # Carry cost
                    carry = cfg.daily_carry_cost

                    current_trade.pnl_underlying += und_pnl
                    current_trade.pnl_vega += vega_pnl
                    current_trade.pnl_gamma += gamma_pnl
                    current_trade.pnl_theta += theta_pnl
                    current_trade.cost_carry += carry

                    daily_pnl += und_pnl + vega_pnl + gamma_pnl + theta_pnl - carry
                    hold_days += 1

            equity *= (1 + daily_pnl)
            self.daily_returns.append(daily_pnl)
            self.daily_positions.append(pos)
            self.daily_equity.append(equity)
            self.dates.append(dates[i])

        # Close any open trade at end
        if current_trade is not None:
            current_trade.exit_date = dates[-1]
            current_trade.exit_z = z_scores[-1]
            current_trade.smh_iv_exit = smh_iv[-1]
            current_trade.qqq_iv_exit = qqq_iv[-1]
            current_trade.hold_days = hold_days
            current_trade.pnl_total = (
                current_trade.pnl_underlying
                + current_trade.pnl_vega
                + current_trade.pnl_gamma
                + current_trade.pnl_theta
                - current_trade.cost_slippage
                - current_trade.cost_carry
            )
            self.trades.append(current_trade)

        return self

    # ----------------------------------------------------------
    # REPORTING
    # ----------------------------------------------------------

    def portfolio_metrics(self) -> dict:
        rets = np.array(self.daily_returns)
        equity = np.array(self.daily_equity)
        ann_ret = np.mean(rets) * 252
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dd = np.min(dd) if len(dd) > 0 else 0.0

        wins = [t for t in self.trades if t.pnl_total > 0]
        losses = [t for t in self.trades if t.pnl_total <= 0]
        avg_win = np.mean([t.pnl_total for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_total for t in losses]) if losses else 0.0

        return {
            "Total Return": f"{equity[-1] - 1:.2%}",
            "Annualized Return": f"{ann_ret:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Total Trades": len(self.trades),
            "Win Rate": f"{len(wins) / max(len(self.trades), 1):.1%}",
            "Avg Win": f"{avg_win:.4f}",
            "Avg Loss": f"{avg_loss:.4f}",
            "Profit Factor": f"{abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "inf",
            "Avg Hold Days": f"{np.mean([t.hold_days for t in self.trades]):.1f}" if self.trades else "0",
            "Days in Market": f"{np.count_nonzero(self.daily_positions)} / {len(self.daily_positions)}",
        }

    def trade_log_df(self) -> pd.DataFrame:
        records = []
        for t in self.trades:
            records.append({
                "Trade #": t.trade_id,
                "Direction": t.direction,
                "Entry Date": t.entry_date,
                "Exit Date": t.exit_date,
                "Hold Days": t.hold_days,
                "Entry Z": round(t.entry_z, 3),
                "Exit Z": round(t.exit_z, 3),
                "SMH IV Entry": f"{t.smh_iv_entry:.1%}",
                "QQQ IV Entry": f"{t.qqq_iv_entry:.1%}",
                "SMH IV Exit": f"{t.smh_iv_exit:.1%}",
                "QQQ IV Exit": f"{t.qqq_iv_exit:.1%}",
                "P&L Underlying": round(t.pnl_underlying, 5),
                "P&L Vega": round(t.pnl_vega, 5),
                "P&L Gamma": round(t.pnl_gamma, 5),
                "P&L Theta": round(t.pnl_theta, 5),
                "Cost Slippage": round(t.cost_slippage, 5),
                "Cost Carry": round(t.cost_carry, 5),
                "Total P&L": round(t.pnl_total, 5),
            })
        return pd.DataFrame(records)

    def equity_curve_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": self.dates,
            "Equity": self.daily_equity,
            "Daily Return": self.daily_returns,
            "Position": self.daily_positions,
        })

    def print_report(self):
        print("\n" + "=" * 70)
        print("  SMH/QQQ RELATIVE VOLATILITY PAIRS STRATEGY — BACKTEST REPORT")
        print("=" * 70)

        print("\n--- Portfolio Metrics ---")
        for k, v in self.portfolio_metrics().items():
            print(f"  {k:.<30} {v}")

        print("\n--- P&L Attribution (all trades) ---")
        if self.trades:
            total_und = sum(t.pnl_underlying for t in self.trades)
            total_vega = sum(t.pnl_vega for t in self.trades)
            total_gamma = sum(t.pnl_gamma for t in self.trades)
            total_theta = sum(t.pnl_theta for t in self.trades)
            total_slip = sum(t.cost_slippage for t in self.trades)
            total_carry = sum(t.cost_carry for t in self.trades)
            print(f"  Underlying (ETF spread):  {total_und:+.4f}")
            print(f"  Vega (IV changes):        {total_vega:+.4f}")
            print(f"  Gamma (realized moves):   {total_gamma:+.4f}")
            print(f"  Theta (time decay):       {total_theta:+.4f}")
            print(f"  Slippage (entry+exit):    {-total_slip:+.4f}")
            print(f"  Carry (daily drag):       {-total_carry:+.4f}")
            print(f"  {'─' * 40}")
            net = total_und + total_vega + total_gamma + total_theta - total_slip - total_carry
            print(f"  NET:                      {net:+.4f}")

        print("\n--- Trade Log ---")
        log_df = self.trade_log_df()
        if not log_df.empty:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            print(log_df.to_string(index=False))
        else:
            print("  No trades executed.")

        print("\n" + "=" * 70)


# ============================================================
# LIVE SIGNAL (current positioning based on latest data)
# ============================================================

def get_live_signal(config: StrategyConfig) -> dict:
    """Fetch latest data and compute current signal for real-time monitoring."""
    df = load_market_data(config)
    df = compute_signal(df, config)
    latest = df.iloc[-1]
    z = latest["z_score"]

    if z > config.entry_thresh + config.reentry_buffer:
        signal = "SHORT_SMH_VOL"
    elif z < -(config.entry_thresh + config.reentry_buffer):
        signal = "LONG_SMH_VOL"
    else:
        signal = "NEUTRAL"

    return {
        "date": latest.name,
        "z_score": round(z, 3),
        "smh_iv": f"{latest['smh_iv']:.1%}",
        "qqq_iv": f"{latest['qqq_iv']:.1%}",
        "signal": signal,
        "smh_price": f"${latest['SMH']:.2f}",
        "qqq_price": f"${latest['QQQ']:.2f}",
    }


# ============================================================
# MAIN
# ============================================================

def main():
    config = StrategyConfig()

    print("Loading market data from Yahoo Finance...")
    df = load_market_data(config)
    print(f"  Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")

    print("Computing signal...")
    df = compute_signal(df, config)
    print(f"  Z-score range: {df['z_score'].min():.2f} to {df['z_score'].max():.2f}")

    print("Running backtest...")
    bt = Backtester(config)
    bt.run(df)
    bt.print_report()

    # Save outputs
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_output")
    os.makedirs(output_dir, exist_ok=True)

    trade_log_path = os.path.join(output_dir, "trade_log.csv")
    bt.trade_log_df().to_csv(trade_log_path, index=False)

    equity_path = os.path.join(output_dir, "equity_curve.csv")
    bt.equity_curve_df().to_csv(equity_path, index=False)

    print(f"\n  Trade log saved to: {trade_log_path}")
    print(f"  Equity curve saved to: {equity_path}")

    # Live signal
    print("\n--- Current Live Signal ---")
    live = get_live_signal(config)
    for k, v in live.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
