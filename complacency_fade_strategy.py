"""
Complacency Fade Strategy: AI Semiconductor Volatility
=======================================================
Trades the recurring pattern of AI semiconductor stocks (NVDA, AVGO, TSM / SMH):
  - Long quiet drift upward → compresses vol → options get cheap
  - Followed by sharp repricing on earnings, export bans, macro shocks

Signal: 4-component composite (term structure, IV/RV spread, correlation, skew)
Trade:  Short underlying + long ATM straddles when composite ≤ 15th percentile
Capital: $10,000 starting capital — all P&L expressed in USD

Usage:
    python complacency_fade_strategy.py              # full backtest + charts
    python complacency_fade_strategy.py --live       # print today's live signal
    python complacency_fade_strategy.py --mc         # add Monte Carlo (slow, ~5min)

Requirements: pip install yfinance pandas numpy scipy matplotlib
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

CFG = dict(
    # Universe
    constituents     = ["NVDA", "AVGO", "TSM"],
    index_ticker     = "SMH",
    vix_proxy        = "^VXN",          # Nasdaq vol index as IV proxy
    start            = "2021-04-01",
    end              = datetime.today().strftime("%Y-%m-%d"),

    # Signal windows
    short_vol_window = 10,
    long_vol_window  = 63,
    corr_window      = 10,
    skew_window      = 21,
    rank_window      = 252,

    # Signal weights
    w_ts             = 0.35,
    w_ivrv           = 0.30,
    w_corr           = 0.20,
    w_skew           = 0.15,

    # Entry / exit thresholds (percentile)
    entry_pctile     = 15,
    exit_pctile      = 50,

    # Trade parameters
    max_hold_days    = 42,
    min_hold_days    = 7,
    underlying_alloc = 0.50,          # 50% of capital in underlying short
    straddle_alloc   = 0.50,          # 50% of capital in straddles

    # Cost model
    stock_slip_rt    = 0.001,         # 10 bps round-trip
    option_slip_rt   = 0.012,         # 1.2% of premium per leg round-trip
    theta_daily      = 0.00035,       # 0.035% per day on straddle notional

    # Gamma model params
    gamma_multiplier = 25.0,          # lower to reflect real straddle payoff structure
    vol_move_beta    = 0.40,          # sensitivity of straddle to IV changes

    # Capital
    starting_capital = 10_000,        # USD

    # Monte Carlo
    mc_paths         = 1000,
    mc_seed          = 42,
)


# ─────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers via yfinance."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers].dropna(how="all")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.ffill().dropna()



# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualised realised volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0–100) of each value vs its past 'window' obs."""
    def _rank(x):
        if len(x) < 2:
            return np.nan
        return stats.percentileofscore(x[:-1], x[-1], kind="rank")
    return series.rolling(window + 1, min_periods=window // 2).apply(_rank, raw=True)


def build_features(prices: pd.DataFrame, vxn: pd.Series, cfg: dict) -> pd.DataFrame:
    """Compute the four complacency features and composite score."""
    sw  = cfg["short_vol_window"]
    lw  = cfg["long_vol_window"]
    cw  = cfg["corr_window"]
    skw = cfg["skew_window"]
    rw  = cfg["rank_window"]
    tickers = cfg["constituents"]

    # Returns
    rets = prices[tickers].pct_change()

    # 1. Term-structure ratio  (avg across constituents)
    ts_ratios = pd.Series(dtype=float)
    for t in tickers:
        rv_short = realized_vol(rets[t], sw)
        rv_long  = realized_vol(rets[t], lw)
        ts_ratios = ts_ratios.add(rv_short / rv_long.replace(0, np.nan), fill_value=0)
    ts_ratio = ts_ratios / len(tickers)

    # 2. IV / RV spread  — use VXN as IV proxy, scaled by avg basket vol ratio
    avg_rv = sum(realized_vol(rets[t], lw) for t in tickers) / len(tickers)
    vxn_aligned = vxn.reindex(prices.index).ffill()
    iv_proxy = vxn_aligned / 100.0
    ivrv_spread = iv_proxy / avg_rv.replace(0, np.nan)

    # 3. Average pairwise correlation
    rolling_corr = pd.Series(index=prices.index, dtype=float)
    pairs = [(tickers[i], tickers[j])
             for i in range(len(tickers)) for j in range(i+1, len(tickers))]
    for i in range(cw, len(prices)):
        window_rets = rets.iloc[i-cw:i][tickers]
        corr_mat = window_rets.corr()
        avg_c = np.mean([corr_mat.loc[a, b] for a, b in pairs])
        rolling_corr.iloc[i] = avg_c

    # 4. Average return skewness
    rolling_skew = sum(
        rets[t].rolling(skw).skew() for t in tickers
    ) / len(tickers)

    # Percentile ranks
    ts_pct   = percentile_rank(ts_ratio,   rw)
    ivrv_pct = percentile_rank(ivrv_spread, rw)
    corr_pct = percentile_rank(rolling_corr, rw)
    skew_pct = percentile_rank(rolling_skew, rw)

    # Weighted composite
    composite = (
        cfg["w_ts"]   * ts_pct +
        cfg["w_ivrv"] * ivrv_pct +
        cfg["w_corr"] * corr_pct +
        cfg["w_skew"] * skew_pct
    )
    composite_pct = percentile_rank(composite, rw)

    feat = pd.DataFrame({
        "ts_ratio":       ts_ratio,
        "ivrv_spread":    ivrv_spread,
        "avg_corr":       rolling_corr,
        "avg_skew":       rolling_skew,
        "ts_pct":         ts_pct,
        "ivrv_pct":       ivrv_pct,
        "corr_pct":       corr_pct,
        "skew_pct":       skew_pct,
        "composite":      composite,
        "composite_pct":  composite_pct,
    }, index=prices.index)

    feat["regime"] = "NEUTRAL"
    feat.loc[feat["composite_pct"] <= cfg["entry_pctile"], "regime"] = "COMPLACENT"
    feat.loc[feat["composite_pct"] >= 75, "regime"] = "FEAR"

    return feat


# ─────────────────────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────────────────────

def run_backtest(prices: pd.DataFrame, feat: pd.DataFrame,
                 index_prices: pd.Series, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Simulate the strategy day-by-day.
    Returns (equity_curve_df, trade_log).
    """
    idx_ret = index_prices.pct_change().reindex(feat.index)

    # Basket return (equal-weight constituents)
    basket_rets = prices[cfg["constituents"]].pct_change().mean(axis=1).reindex(feat.index)

    equity = cfg["starting_capital"]
    in_trade  = False
    entry_day = None
    entry_idx_price = None
    hold_count = 0
    entry_iv = None

    daily_records = []
    trades = []

    total_slip = cfg["stock_slip_rt"] + cfg["option_slip_rt"]

    for i, date in enumerate(feat.index):
        regime = feat["composite_pct"].iloc[i]
        regime_label = feat["regime"].iloc[i]

        daily_pnl = 0.0
        action = "HOLD" if in_trade else "FLAT"

        # ── ENTRY ──────────────────────────────────────────────
        if not in_trade and i > 0 and feat["regime"].iloc[i-1] == "COMPLACENT":
            in_trade = True
            entry_day = date
            hold_count = 0
            entry_iv = feat["ivrv_spread"].iloc[i]
            entry_equity = equity
            daily_pnl -= total_slip / 2 * equity   # entry slippage
            action = "ENTRY"

        # ── DAILY P&L (if in trade) ─────────────────────────────
        if in_trade and action != "ENTRY":
            hold_count += 1
            ix_ret = idx_ret.iloc[i] if not np.isnan(idx_ret.iloc[i]) else 0.0
            bk_ret = basket_rets.iloc[i] if not np.isnan(basket_rets.iloc[i]) else 0.0

            # Underlying short
            pnl_underlying = -cfg["underlying_alloc"] * (ix_ret * 0.5 + bk_ret * 0.5) * equity

            # Theta
            pnl_theta = -cfg["theta_daily"] * cfg["straddle_alloc"] * equity

            # Gamma: profit when realized move > implied daily vol
            current_iv = feat["ivrv_spread"].iloc[i]
            implied_daily_var = (current_iv * 0.25) ** 2 / 252  # rough ATM implied daily variance
            realized_sq = max(ix_ret, bk_ret) ** 2
            pnl_gamma = (cfg["straddle_alloc"] * equity *
                         max(0, realized_sq - implied_daily_var) *
                         cfg["gamma_multiplier"])

            # Vol move (vega)
            if i > 0 and not np.isnan(feat["ivrv_spread"].iloc[i-1]):
                iv_chg = feat["ivrv_spread"].iloc[i] - feat["ivrv_spread"].iloc[i-1]
            else:
                iv_chg = 0.0
            pnl_vega = cfg["straddle_alloc"] * equity * cfg["vol_move_beta"] * iv_chg

            daily_pnl += pnl_underlying + pnl_theta + pnl_gamma + pnl_vega

        # ── EXIT ────────────────────────────────────────────────
        should_exit = False
        exit_reason = ""
        if in_trade and action not in ("ENTRY",):
            normalized = (regime >= cfg["exit_pctile"] and hold_count >= cfg["min_hold_days"])
            max_hold   = hold_count >= cfg["max_hold_days"]
            fear_flip  = regime_label == "FEAR"

            if normalized:  should_exit, exit_reason = True, "NORMALIZED"
            if max_hold:    should_exit, exit_reason = True, "MAX_HOLD"
            if fear_flip:   should_exit, exit_reason = True, "FEAR_FLIP"

        if should_exit:
            daily_pnl -= total_slip / 2 * equity   # exit slippage
            trade_ret  = (equity / entry_equity) - 1.0
            trades.append({
                "entry_date": entry_day,
                "exit_date":  date,
                "hold_days":  hold_count,
                "exit_reason":exit_reason,
                "trade_return": trade_ret,
            })
            in_trade = False
            action   = "EXIT"

        equity = max(equity + daily_pnl, 1.0)

        daily_records.append({
            "date":     date,
            "equity":   equity,
            "pnl":      daily_pnl,
            "regime":   regime_label,
            "composite_pct": regime,
            "in_trade": in_trade or action in ("ENTRY", "EXIT"),
            "action":   action,
        })

    df = pd.DataFrame(daily_records).set_index("date")
    return df, trades


# ─────────────────────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────

def calc_metrics(equity: pd.Series, trades: list[dict], years: float) -> dict:
    daily_rets = equity.pct_change().dropna()
    total_ret  = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret    = (1 + total_ret) ** (1 / years) - 1
    sharpe     = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                  if daily_rets.std() > 0 else 0)
    peak       = equity.cummax()
    drawdown   = (equity - peak) / peak
    max_dd     = drawdown.min()

    trade_rets = [t["trade_return"] for t in trades]
    wins       = [r for r in trade_rets if r > 0]
    losses     = [r for r in trade_rets if r <= 0]
    win_rate   = len(wins) / len(trades) if trades else 0
    profit_factor = (sum(wins) / abs(sum(losses))
                     if losses and sum(losses) != 0 else np.inf)
    avg_hold   = np.mean([t["hold_days"] for t in trades]) if trades else 0

    return dict(
        starting_capital = equity.iloc[0],
        final_value      = equity.iloc[-1],
        net_profit       = equity.iloc[-1] - equity.iloc[0],
        total_return     = total_ret,
        ann_return       = ann_ret,
        sharpe           = sharpe,
        max_drawdown     = max_dd,
        max_drawdown_usd = max_dd * equity.iloc[0],
        n_trades         = len(trades),
        win_rate         = win_rate,
        profit_factor    = profit_factor,
        avg_hold         = avg_hold,
        avg_win          = np.mean(wins)   if wins   else 0,
        avg_loss         = np.mean(losses) if losses else 0,
    )


# ─────────────────────────────────────────────────────────────
#  MONTE CARLO
# ─────────────────────────────────────────────────────────────

def run_monte_carlo(cfg: dict, n_paths: int = 1000,
                    n_days: int = 1260, seed: int = 42) -> pd.DataFrame:
    """
    Simulate strategy on synthetic paths with:
    - Stochastic (Ornstein-Uhlenbeck) volatility
    - Student-t fat tails (df=4.2)
    - Correlation clustering
    """
    rng = np.random.default_rng(seed)
    results = []

    # OU vol params (calibrated from real data)
    vol_mean, vol_speed, vol_vol = 0.35, 0.15, 0.10
    corr_mean, corr_speed = 0.55, 0.25
    t_df = 4.2

    for _ in range(n_paths):
        # Simulate volatility path (OU)
        vol = np.zeros(n_days)
        vol[0] = vol_mean
        corr_path = np.zeros(n_days)
        corr_path[0] = corr_mean
        for t in range(1, n_days):
            dW_vol  = rng.standard_normal()
            dW_corr = rng.standard_normal()
            vol[t]  = max(0.05, vol[t-1] + vol_speed*(vol_mean - vol[t-1])/252
                          + vol_vol/np.sqrt(252) * dW_vol)
            corr_path[t] = np.clip(
                corr_path[t-1] + corr_speed*(corr_mean - corr_path[t-1])/252
                + 0.15/np.sqrt(252) * dW_corr, 0.1, 0.99)

        # Simulate index returns (fat-tailed)
        t_draws = rng.standard_t(t_df, size=n_days)
        idx_rets = vol / np.sqrt(252) * t_draws / np.sqrt(t_df / (t_df - 2))

        # Build synthetic features
        rv_short = pd.Series(idx_rets).rolling(cfg["short_vol_window"]).std() * np.sqrt(252)
        rv_long  = pd.Series(idx_rets).rolling(cfg["long_vol_window"]).std() * np.sqrt(252)
        ts_ratio = rv_short / rv_long.replace(0, np.nan)
        ivrv     = pd.Series(vol) / rv_long.replace(0, np.nan)
        corr_s   = pd.Series(corr_path)
        skew_s   = pd.Series(idx_rets).rolling(cfg["skew_window"]).skew()

        rw = cfg["rank_window"]
        ts_pct   = percentile_rank(ts_ratio, rw)
        ivrv_pct = percentile_rank(ivrv, rw)
        corr_pct = percentile_rank(corr_s, rw)
        skew_pct = percentile_rank(skew_s, rw)

        composite = (cfg["w_ts"]   * ts_pct +
                     cfg["w_ivrv"] * ivrv_pct +
                     cfg["w_corr"] * corr_pct +
                     cfg["w_skew"] * skew_pct)
        comp_pct  = percentile_rank(composite, rw)

        # Simulate trades
        equity = cfg["starting_capital"]
        in_trade = False
        hold_count = 0
        entry_equity = 1.0
        trade_rets_sim = []
        eq_curve = []
        total_slip = cfg["stock_slip_rt"] + cfg["option_slip_rt"]

        for i in range(n_days):
            cp = comp_pct.iloc[i] if not np.isnan(comp_pct.iloc[i]) else 50
            ret_i = idx_rets[i]

            daily_pnl = 0.0

            if not in_trade and i > 0:
                prev_cp = comp_pct.iloc[i-1] if not np.isnan(comp_pct.iloc[i-1]) else 50
                if prev_cp <= cfg["entry_pctile"]:
                    in_trade = True
                    hold_count = 0
                    entry_equity = equity
                    daily_pnl -= total_slip / 2 * equity

            if in_trade:
                hold_count += 1
                iv_now = ivrv.iloc[i] if not np.isnan(ivrv.iloc[i]) else 1.0
                impl_dv = (iv_now * 0.25) ** 2 / 252
                pnl_u   = -cfg["underlying_alloc"] * ret_i * equity
                pnl_t   = -cfg["theta_daily"] * cfg["straddle_alloc"] * equity
                pnl_g   = (cfg["straddle_alloc"] * equity *
                           max(0, ret_i**2 - impl_dv) * cfg["gamma_multiplier"])
                iv_chg  = (ivrv.iloc[i] - ivrv.iloc[i-1]
                           if i > 0 and not np.isnan(ivrv.iloc[i-1]) else 0)
                pnl_vg  = cfg["straddle_alloc"] * equity * cfg["vol_move_beta"] * iv_chg
                daily_pnl += pnl_u + pnl_t + pnl_g + pnl_vg

                normalized = (cp >= cfg["exit_pctile"] and hold_count >= cfg["min_hold_days"])
                max_hold   = hold_count >= cfg["max_hold_days"]
                if normalized or max_hold:
                    daily_pnl -= total_slip / 2 * equity
                    trade_rets_sim.append(equity / entry_equity - 1)
                    in_trade = False

            equity = max(equity + daily_pnl, 1.0)
            eq_curve.append(equity)

        eq_s = pd.Series(eq_curve)
        dr   = eq_s.pct_change().dropna()
        tot  = eq_s.iloc[-1] / eq_s.iloc[0] - 1
        ann  = (1 + tot) ** (252 / n_days) - 1
        sh   = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        peak = eq_s.cummax()
        mdd  = ((eq_s - peak) / peak).min()
        wr   = (sum(1 for r in trade_rets_sim if r > 0) /
                len(trade_rets_sim) if trade_rets_sim else 0)
        wins_  = [r for r in trade_rets_sim if r > 0]
        losses_= [r for r in trade_rets_sim if r <= 0]
        pf     = (sum(wins_) / abs(sum(losses_))
                  if losses_ and sum(losses_) != 0 else 10.0)

        results.append({
            "total_return":  tot,
            "ann_return":    ann,
            "sharpe":        sh,
            "max_drawdown":  mdd,
            "n_trades":      len(trade_rets_sim),
            "win_rate":      wr,
            "profit_factor": min(pf, 20.0),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "border":   "#30363d",
    "green":    "#3fb950",
    "red":      "#f85149",
    "amber":    "#d29922",
    "blue":     "#58a6ff",
    "purple":   "#bc8cff",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}

def apply_theme(fig, axes):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["subtext"])
        ax.yaxis.label.set_color(PALETTE["subtext"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.title.set_color(PALETTE["text"])


def plot_backtest(equity_df: pd.DataFrame, trades: list[dict],
                  feat: pd.DataFrame, metrics: dict, out_path: str):

    fig = plt.figure(figsize=(16, 12), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                             left=0.07, right=0.97, top=0.93, bottom=0.07)

    ax_eq  = fig.add_subplot(gs[0, :])   # equity curve full width
    ax_dd  = fig.add_subplot(gs[1, 0])   # drawdown
    ax_sig = fig.add_subplot(gs[1, 1])   # composite signal
    ax_wf  = fig.add_subplot(gs[2, 0])   # trade waterfall
    ax_sc  = fig.add_subplot(gs[2, 1])   # scatter: hold vs return

    axes = [ax_eq, ax_dd, ax_sig, ax_wf, ax_sc]
    apply_theme(fig, axes)

    eq = equity_df["equity"]

    # ── Equity curve ──────────────────────────────────────────
    ax_eq.plot(eq.index, eq.values, color=PALETTE["green"], lw=1.8, label="Strategy")
    # shade in-trade periods
    in_trade_mask = equity_df["in_trade"]
    for i in range(len(equity_df)):
        if in_trade_mask.iloc[i]:
            ax_eq.axvspan(equity_df.index[i], equity_df.index[min(i+1, len(equity_df)-1)],
                          alpha=0.08, color=PALETTE["blue"], lw=0)
    ax_eq.set_title("Portfolio Equity Curve  (starting capital $10,000)", fontsize=11, pad=8)
    ax_eq.set_ylabel("Portfolio Value ($)", color=PALETTE["subtext"])
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    # stats box
    stats_txt = (f"Start: ${metrics['starting_capital']:,.0f}   "
                 f"End: ${metrics['final_value']:,.0f}   "
                 f"Net P&L: ${metrics['net_profit']:+,.0f}   "
                 f"Ann: {metrics['ann_return']*100:+.1f}%   "
                 f"Sharpe: {metrics['sharpe']:.2f}   "
                 f"MaxDD: {metrics['max_drawdown']*100:.1f}%   "
                 f"Trades: {metrics['n_trades']}   "
                 f"WinRate: {metrics['win_rate']*100:.0f}%")
    ax_eq.text(0.01, 0.97, stats_txt, transform=ax_eq.transAxes,
               color=PALETTE["text"], fontsize=8.5, va="top",
               bbox=dict(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"], pad=4))

    # ── Drawdown ───────────────────────────────────────────────
    peak    = eq.cummax()
    dd_usd  = eq - peak          # dollar drawdown (always ≤ 0)
    ax_dd.fill_between(dd_usd.index, dd_usd.values, 0, color=PALETTE["red"], alpha=0.7)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_dd.set_title("Drawdown ($)", fontsize=10)
    ax_dd.set_ylabel("$", color=PALETTE["subtext"])

    # ── Composite signal ──────────────────────────────────────
    comp = feat["composite_pct"].reindex(equity_df.index)
    ax_sig.plot(comp.index, comp.values, color=PALETTE["purple"], lw=1.2, alpha=0.9)
    ax_sig.axhline(CFG["entry_pctile"], color=PALETTE["green"], lw=1, ls="--",
                   label=f"Entry ({CFG['entry_pctile']}th)")
    ax_sig.axhline(CFG["exit_pctile"],  color=PALETTE["amber"], lw=1, ls="--",
                   label=f"Exit ({CFG['exit_pctile']}th)")
    ax_sig.fill_between(comp.index, comp.values, CFG["entry_pctile"],
                        where=(comp <= CFG["entry_pctile"]),
                        color=PALETTE["green"], alpha=0.25)
    ax_sig.set_title("Complacency Score (composite percentile)", fontsize=10)
    ax_sig.set_ylim(0, 100)
    ax_sig.legend(fontsize=7, labelcolor=PALETTE["subtext"],
                  facecolor=PALETTE["panel"], edgecolor=PALETTE["border"])

    # ── Trade waterfall ────────────────────────────────────────
    if trades:
        trade_df = pd.DataFrame(trades).sort_values("entry_date")
        colors   = [PALETTE["green"] if r > 0 else PALETTE["red"]
                    for r in trade_df["trade_return"]]
        bars = ax_wf.bar(range(len(trade_df)), trade_df["trade_return"] * 100,
                         color=colors, width=0.7, edgecolor=PALETTE["bg"], linewidth=0.5)
        ax_wf.axhline(0, color=PALETTE["border"], lw=0.8)
        ax_wf.set_title("Individual Trade Returns (%)", fontsize=10)
        ax_wf.set_xlabel("Trade #", color=PALETTE["subtext"])
        ax_wf.set_ylabel("%", color=PALETTE["subtext"])
        ax_wf.set_xticks(range(len(trade_df)))
        ax_wf.set_xticklabels([str(i+1) for i in range(len(trade_df))], fontsize=7)
        # label each bar
        for bar, val in zip(bars, trade_df["trade_return"] * 100):
            ypos = val + (0.5 if val >= 0 else -1.5)
            ax_wf.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:.1f}",
                       ha="center", va="bottom", fontsize=6, color=PALETTE["text"])

    # ── Hold vs return scatter ─────────────────────────────────
    if trades:
        xs = [t["hold_days"] for t in trades]
        ys = [t["trade_return"] * 100 for t in trades]
        cols = [PALETTE["green"] if y > 0 else PALETTE["red"] for y in ys]
        ax_sc.scatter(xs, ys, c=cols, s=60, edgecolors=PALETTE["border"],
                      linewidths=0.5, alpha=0.85)
        ax_sc.axhline(0, color=PALETTE["border"], lw=0.8)
        ax_sc.set_title("Hold Days vs Trade Return", fontsize=10)
        ax_sc.set_xlabel("Hold Days", color=PALETTE["subtext"])
        ax_sc.set_ylabel("Return (%)", color=PALETTE["subtext"])

    fig.suptitle("Complacency Fade Strategy — AI Semiconductor Volatility",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Backtest chart saved → {out_path}")


def plot_monte_carlo(mc_df: pd.DataFrame, historical_metrics: dict, out_path: str):

    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                             left=0.07, right=0.97, top=0.92, bottom=0.08)

    metrics_plot = [
        ("sharpe",        "Sharpe Ratio",        PALETTE["blue"]),
        ("total_return",  "Total Return",         PALETTE["green"]),
        ("max_drawdown",  "Max Drawdown",         PALETTE["red"]),
        ("n_trades",      "# Trades",             PALETTE["purple"]),
        ("win_rate",      "Win Rate",             PALETTE["amber"]),
        ("profit_factor", "Profit Factor",        PALETTE["blue"]),
    ]

    hist_vals = {
        "sharpe":        historical_metrics["sharpe"],
        "total_return":  historical_metrics["total_return"],
        "max_drawdown":  historical_metrics["max_drawdown"],
        "n_trades":      historical_metrics["n_trades"],
        "win_rate":      historical_metrics["win_rate"],
        "profit_factor": historical_metrics["profit_factor"],
    }

    for idx, (col, title, color) in enumerate(metrics_plot):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        apply_theme(fig, [ax])

        data = mc_df[col].dropna()
        ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor=PALETTE["bg"], lw=0.3)
        ax.axvline(data.median(), color=PALETTE["text"], lw=1.5, ls="--", label="Median")
        hv = hist_vals[col]
        ax.axvline(hv, color=PALETTE["amber"], lw=2.0, label=f"Historical ({hv:.2f})")

        p5  = np.percentile(data, 5)
        p95 = np.percentile(data, 95)
        ax.axvspan(p5, p95, alpha=0.12, color=color)

        # probability annotations
        if col == "sharpe":
            prob = (data > 0).mean()
            ax.text(0.97, 0.95, f"P(>0): {prob*100:.1f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    color=PALETTE["text"], fontsize=8)

        ax.set_title(title, fontsize=10, pad=6)
        ax.legend(fontsize=7, labelcolor=PALETTE["subtext"],
                  facecolor=PALETTE["panel"], edgecolor=PALETTE["border"])

    fig.suptitle(f"Monte Carlo Simulation — {len(mc_df):,} Paths  |  "
                 "Complacency Fade Strategy",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", y=0.97)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Monte Carlo chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────
#  LIVE SIGNAL REPORT
# ─────────────────────────────────────────────────────────────

def print_live_signal(feat: pd.DataFrame, prices: pd.DataFrame):
    last = feat.iloc[-1]
    date = feat.index[-1].strftime("%Y-%m-%d")

    regime_color = {
        "COMPLACENT": "\033[92m",   # green
        "FEAR":       "\033[91m",   # red
        "NEUTRAL":    "\033[93m",   # yellow
    }
    rc = regime_color.get(last["regime"], "")
    RESET = "\033[0m"

    print("\n" + "─" * 60)
    print(f"  COMPLACENCY FADE STRATEGY — LIVE SIGNAL")
    print(f"  {date}")
    print("─" * 60)
    print(f"  Regime:            {rc}{last['regime']}{RESET}")
    print(f"  Composite Pctile:  {last['composite_pct']:.1f}th")
    print(f"  Term-Structure:    {last['ts_ratio']:.3f}  (pctile: {last['ts_pct']:.0f})")
    print(f"  IV/RV Spread:      {last['ivrv_spread']:.3f}  (pctile: {last['ivrv_pct']:.0f})")
    print(f"  Avg Correlation:   {last['avg_corr']:.3f}  (pctile: {last['corr_pct']:.0f})")
    print(f"  Avg Skew:          {last['avg_skew']:.3f}  (pctile: {last['skew_pct']:.0f})")
    print("─" * 60)
    if last["composite_pct"] <= CFG["entry_pctile"]:
        print("  ⚡ SIGNAL ACTIVE — Consider: short SMH + long straddles")
    elif last["composite_pct"] >= 75:
        print("  🔴 FEAR REGIME — No new entries; exit if in trade")
    else:
        print("  ⬜ NEUTRAL — No action")
    print("─" * 60 + "\n")


def print_metrics(metrics: dict, label: str = "Backtest Results"):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Starting Capital: ${metrics['starting_capital']:>10,.2f}")
    print(f"  Final Value:      ${metrics['final_value']:>10,.2f}")
    print(f"  Net Profit:       ${metrics['net_profit']:>+10,.2f}")
    print(f"  Total Return:      {metrics['total_return']*100:>+9.1f}%")
    print(f"  Annual Return:     {metrics['ann_return']*100:>+9.1f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe']:>10.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']*100:>9.1f}%  (${abs(metrics['max_drawdown_usd']):,.2f})")
    print(f"  # Trades:          {metrics['n_trades']:>10}")
    print(f"  Win Rate:          {metrics['win_rate']*100:>9.0f}%")
    print(f"  Profit Factor:     {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Hold:          {metrics['avg_hold']:>9.1f} days")
    print(f"  Avg Win:           {metrics['avg_win']*100:>+9.1f}%")
    print(f"  Avg Loss:          {metrics['avg_loss']*100:>+9.1f}%")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Complacency Fade Strategy")
    parser.add_argument("--live", action="store_true", help="Print today's live signal only")
    parser.add_argument("--mc",   action="store_true", help="Run Monte Carlo simulation")
    args = parser.parse_args()

    print("\n📡  Fetching market data …")
    tickers = CFG["constituents"] + [CFG["index_ticker"], CFG["vix_proxy"]]
    all_prices = fetch_prices(tickers, CFG["start"], CFG["end"])

    constituent_prices = all_prices[CFG["constituents"]]
    index_prices       = all_prices[CFG["index_ticker"]]
    vxn_series         = all_prices[CFG["vix_proxy"]]

    print("🔬  Computing features …")
    feat = build_features(constituent_prices, vxn_series, CFG)

    if args.live:
        print_live_signal(feat, constituent_prices)
        return

    print_live_signal(feat, constituent_prices)

    print("📈  Running backtest …")
    equity_df, trades = run_backtest(constituent_prices, feat, index_prices, CFG)

    years = (feat.index[-1] - feat.index[0]).days / 365.25
    metrics = calc_metrics(equity_df["equity"], trades, years)
    print_metrics(metrics, "Historical Backtest (SMH Universe)")

    print("\n📊  Generating backtest charts …")
    plot_backtest(equity_df, trades, feat, metrics,
                  "outputs/complacency_fade_backtest.png")

    if args.mc:
        print(f"\n🎲  Running Monte Carlo ({CFG['mc_paths']:,} paths) — may take a few minutes …")
        mc_df = run_monte_carlo(CFG, n_paths=CFG["mc_paths"], seed=CFG["mc_seed"])

        print("\n  Monte Carlo Summary")
        print(f"  Median Sharpe:         {mc_df['sharpe'].median():.2f}")
        print(f"  P(Sharpe > 0):         {(mc_df['sharpe'] > 0).mean()*100:.1f}%")
        print(f"  P(Sharpe > 0.5):       {(mc_df['sharpe'] > 0.5).mean()*100:.1f}%")
        print(f"  P(positive return):    {(mc_df['total_return'] > 0).mean()*100:.1f}%")
        print(f"  5th pctile return:     {mc_df['total_return'].quantile(0.05)*100:.1f}%")
        print(f"  95th pctile return:    {mc_df['total_return'].quantile(0.95)*100:.1f}%")
        print(f"  Worst max drawdown:    {mc_df['max_drawdown'].min()*100:.1f}%")

        print("\n📊  Generating Monte Carlo charts …")
        plot_monte_carlo(mc_df, metrics, "outputs/complacency_fade_mc.png")
        mc_df.to_csv("outputs/monte_carlo_results.csv", index=False)
        print("  ✓ MC results saved → monte_carlo_results.csv")


if __name__ == "__main__":
    main()
