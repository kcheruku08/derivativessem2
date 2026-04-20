"""
Vol Surface Regime Backtester
=============================
Uses volatility surface features (term structure slope, IV-RV spread, skew)
as regime signals to trade BOTH the underlying AND options.

Core insight: The vol surface SHAPE tells you about market regime (fear vs
complacency). Instead of trading vol convergence, we USE vol surface data
to time directional + vol trades together.

Strategy:
---------
UNIVERSE: SMH (index), NVDA, AVGO, TSM (constituents)

SIGNALS (from vol surface):
1. Term Structure Slope: RV_short / RV_long — backwardation = fear
2. IV-RV Spread: proxy_IV / RV — wide = options overpriced
3. Realized Skew: rolling skewness of returns — negative = crash risk

TRADES:
- FEAR REGIME (backwardation + wide IV-RV):
    * BUY underlying (mean-reversion after panic)
    * SELL vol (expensive, collect theta)

- COMPLACENT REGIME (steep contango + narrow IV-RV):
    * SHORT underlying (or reduce)
    * BUY vol (cheap protection, pay theta)

- NEUTRAL:
    * Flat, collect nothing, wait
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

YF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yfinance_cache_regime")
os.makedirs(YF_CACHE_DIR, exist_ok=True)
try:
    yf.set_tz_cache_location(YF_CACHE_DIR)
except Exception:
    pass

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    backtest_years: int = 5
    tickers: tuple = ("SMH", "NVDA", "AVGO", "TSM")
    index_ticker: str = "SMH"

    # Vol surface feature windows
    rv_short_window: int = 10       # short-term realized vol
    rv_long_window: int = 63        # long-term realized vol
    iv_proxy_window: int = 21       # IV proxy from RV + premium
    skew_window: int = 21           # realized skew lookback

    # Regime thresholds (composite score based)
    fear_composite_pctile: float = 75    # not used (no fear trades)
    complacent_composite_pctile: float = 15  # only extreme complacency

    # Position sizing
    underlying_weight: float = 0.5
    options_weight: float = 0.5

    # Holding rules
    min_hold_days: int = 7
    max_hold_days: int = 42          # longer hold to capture full moves

    # Signal lookback
    signal_lookback: int = 252       # 1 year — original best performer

    # Multi-universe: run same signal on additional ETFs
    extra_universes: tuple = (
        ("SOXX", ("SOXX", "NVDA", "AVGO", "TSM")),  # broader semi ETF
        ("XLK", ("XLK", "NVDA", "AAPL", "MSFT")),   # tech sector
    )

    # Costs
    stock_slippage: float = 0.001    # 10 bps per stock trade
    options_slippage: float = 0.012  # 1.2% per options trade (straddle spread)
    theta_daily_short: float = 0.0004  # daily theta collected when short vol
    theta_daily_long: float = 0.00035  # daily theta paid when long vol


# ============================================================
# DATA LOADING
# ============================================================

def load_data(config: Config) -> pd.DataFrame:
    """Download price data for all tickers."""
    period = f"{config.backtest_years}y"
    frames = {}
    for ticker in config.tickers:
        print(f"  Downloading {ticker}...")
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            raise RuntimeError(f"Failed to download {ticker}")
        frames[ticker] = data["Close"].squeeze()

    # Also get VXN for QQQ IV proxy
    print("  Downloading ^VXN...")
    vxn = yf.download("^VXN", period=period, interval="1d", auto_adjust=True, progress=False)
    if not vxn.empty:
        frames["VXN"] = vxn["Close"].squeeze()

    df = pd.DataFrame(frames).dropna()
    print(f"  Combined dataset: {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ============================================================
# VOL SURFACE FEATURE COMPUTATION
# ============================================================

def compute_vol_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute vol surface features from price data.

    Features per constituent and for the index:
    1. term_structure_ratio: RV_short / RV_long (>1 = backwardation, <1 = contango)
    2. iv_rv_spread: IV_proxy / RV (>1 = vol overpriced)
    3. realized_skew: rolling skewness of returns
    4. cross_correlation: rolling corr between constituents (high = danger)
    """
    constituents = [t for t in config.tickers if t != config.index_ticker]

    # Log returns
    for ticker in config.tickers:
        df[f"{ticker}_ret"] = np.log(df[ticker]).diff()

    # Realized vols
    for ticker in config.tickers:
        ret_col = f"{ticker}_ret"
        df[f"{ticker}_rv_short"] = df[ret_col].rolling(config.rv_short_window).std() * np.sqrt(252)
        df[f"{ticker}_rv_long"] = df[ret_col].rolling(config.rv_long_window).std() * np.sqrt(252)

    # Term structure ratio (for each ticker)
    for ticker in config.tickers:
        df[f"{ticker}_ts_ratio"] = df[f"{ticker}_rv_short"] / df[f"{ticker}_rv_long"].replace(0, np.nan)

    # IV proxy: use VXN as the real market IV for the index
    if "VXN" in df.columns:
        df["SMH_iv_proxy"] = df["VXN"] / 100.0
    else:
        df["SMH_iv_proxy"] = df["SMH_rv_short"] * 1.15

    # For constituents, estimate IV from index IV scaled by vol ratio
    # This captures the actual time-varying premium, not a constant
    for ticker in constituents:
        vol_ratio = df[f"{ticker}_rv_short"] / df[f"{config.index_ticker}_rv_short"].replace(0, np.nan)
        vol_ratio_smooth = vol_ratio.ewm(halflife=21, adjust=False).mean().shift(1).fillna(1.5)
        df[f"{ticker}_iv_proxy"] = df["SMH_iv_proxy"] * vol_ratio_smooth

    # IV-RV spread: how much is IV above current RV (time-varying)
    for ticker in config.tickers:
        iv_col = f"{ticker}_iv_proxy"
        rv_col = f"{ticker}_rv_short"
        if iv_col in df.columns:
            df[f"{ticker}_ivrv"] = df[iv_col] / df[rv_col].replace(0, np.nan)
            df[f"{ticker}_ivrv"] = df[f"{ticker}_ivrv"].clip(0.3, 5.0)

    # Realized skewness
    for ticker in config.tickers:
        ret_col = f"{ticker}_ret"
        df[f"{ticker}_skew"] = df[ret_col].rolling(config.skew_window).skew()

    # Cross-correlation between constituents (pairwise average)
    corr_pairs = []
    for i, t1 in enumerate(constituents):
        for t2 in constituents[i+1:]:
            corr_col = f"corr_{t1}_{t2}"
            df[corr_col] = df[f"{t1}_ret"].rolling(config.rv_short_window).corr(df[f"{t2}_ret"])
            corr_pairs.append(corr_col)

    if corr_pairs:
        df["avg_constituent_corr"] = df[corr_pairs].mean(axis=1)

    # Index-constituent correlation
    for ticker in constituents:
        df[f"corr_{config.index_ticker}_{ticker}"] = (
            df[f"{config.index_ticker}_ret"].rolling(config.rv_short_window).corr(df[f"{ticker}_ret"])
        )

    # Aggregate signals: average term structure and IV-RV across basket
    ts_cols = [f"{t}_ts_ratio" for t in constituents]
    ivrv_cols = [f"{t}_ivrv" for t in constituents if f"{t}_ivrv" in df.columns]

    df["basket_ts_ratio"] = df[ts_cols].mean(axis=1)
    df["basket_ivrv"] = df[ivrv_cols].mean(axis=1)
    df["basket_skew"] = df[[f"{t}_skew" for t in constituents]].mean(axis=1)

    # Also compute index-level features
    df["index_ts_ratio"] = df[f"{config.index_ticker}_ts_ratio"]
    df["index_ivrv"] = df.get(f"{config.index_ticker}_ivrv", df["basket_ivrv"])

    return df


# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regimes(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Classify each day into a regime based on vol surface features.

    Uses a COMPOSITE score combining:
    - Term structure ratio (backwardation = fear)
    - IV/RV spread (high = fear, vol overpriced)
    - Correlation (high = fear, systemic risk)
    - Skew (negative = fear)

    Regimes:
    - FEAR: composite score > 75th pctile → buy dip + sell vol
    - COMPLACENT: composite score < 25th pctile → short + buy vol
    - NEUTRAL: in between
    """
    lookback = config.signal_lookback

    # Individual percentile ranks
    df["ts_pctile"] = df["basket_ts_ratio"].rolling(lookback).rank(pct=True) * 100
    df["ivrv_pctile"] = df["basket_ivrv"].rolling(lookback).rank(pct=True) * 100
    df["corr_pctile"] = df["avg_constituent_corr"].rolling(lookback).rank(pct=True) * 100
    df["skew_pctile"] = (-df["basket_skew"]).rolling(lookback).rank(pct=True) * 100  # negative skew = more fear

    # Composite fear score: weighted average of percentiles
    # Higher = more fear (backwardation + expensive vol + high corr + negative skew)
    df["composite_score"] = (
        0.35 * df["ts_pctile"] +
        0.30 * df["ivrv_pctile"] +
        0.20 * df["corr_pctile"] +
        0.15 * df["skew_pctile"]
    )

    # Rank the composite score itself
    df["composite_pctile"] = df["composite_score"].rolling(lookback).rank(pct=True) * 100

    # Regime classification
    regimes = []
    for i in range(len(df)):
        cp = df["composite_pctile"].iloc[i]

        if pd.isna(cp):
            regimes.append("NEUTRAL")
        elif cp >= config.fear_composite_pctile:
            regimes.append("FEAR")
        elif cp <= config.complacent_composite_pctile:
            regimes.append("COMPLACENT")
        else:
            regimes.append("NEUTRAL")

    df["regime"] = regimes

    # Signal strength
    df["fear_strength"] = df["composite_pctile"] / 100.0
    df["complacent_strength"] = 1.0 - df["fear_strength"]

    return df


# ============================================================
# TRADE STRUCTURES
# ============================================================

@dataclass
class Trade:
    trade_id: int
    regime: str  # "FEAR" or "COMPLACENT"
    entry_date: object = None
    exit_date: object = None
    hold_days: int = 0
    exit_reason: str = ""

    # Entry conditions
    entry_ts_ratio: float = 0.0
    entry_ivrv: float = 0.0
    entry_corr: float = 0.0
    entry_skew: float = 0.0
    entry_smh_price: float = 0.0

    # Exit conditions
    exit_ts_ratio: float = 0.0
    exit_ivrv: float = 0.0
    exit_smh_price: float = 0.0

    # P&L components
    pnl_underlying: float = 0.0
    pnl_theta: float = 0.0
    pnl_gamma: float = 0.0
    pnl_vol_move: float = 0.0
    cost_slippage: float = 0.0
    pnl_total: float = 0.0


# ============================================================
# BACKTESTER
# ============================================================

class VolRegimeBacktester:
    def __init__(self, config: Config):
        self.config = config
        self.trades: list[Trade] = []
        self.daily_returns: list[float] = []
        self.daily_positions: list[str] = []
        self.daily_equity: list[float] = []
        self.dates: list = []

    def run(self, df: pd.DataFrame) -> "VolRegimeBacktester":
        cfg = self.config
        n = len(df)
        idx_ticker = cfg.index_ticker

        position = "FLAT"  # FLAT, LONG_FEAR, SHORT_COMPLACENT
        hold_days = 0
        trade_id = 0
        current_trade: Optional[Trade] = None
        equity = 1.0

        for i in range(1, n):
            regime = df["regime"].iloc[i - 1]  # use prior day's regime (no lookahead)
            daily_pnl = 0.0

            # Current features
            ts_ratio = df["basket_ts_ratio"].iloc[i]
            ivrv = df["basket_ivrv"].iloc[i]
            ts_pctile = df["ts_pctile"].iloc[i]
            ivrv_pctile = df["ivrv_pctile"].iloc[i]
            corr = df["avg_constituent_corr"].iloc[i] if "avg_constituent_corr" in df.columns else 0.5

            # --- EXIT LOGIC ---
            should_exit = False
            exit_reason = ""

            if position != "FLAT":
                hold_days += 1

                # Forced exit on max hold
                if hold_days >= cfg.max_hold_days:
                    should_exit = True
                    exit_reason = "max_hold"
                # Regime flip exit
                elif position == "SHORT_COMPLACENT" and regime != "COMPLACENT" and hold_days >= cfg.min_hold_days:
                    should_exit = True
                    exit_reason = "regime_flip"
                # Signal mean-reversion: exit when composite normalizes
                elif position == "SHORT_COMPLACENT" and hold_days >= cfg.min_hold_days:
                    cp = df["composite_pctile"].iloc[i]
                    if not pd.isna(cp) and cp > 50:
                        should_exit = True
                        exit_reason = "signal_normalized"

            if should_exit and current_trade is not None:
                current_trade.exit_date = df.index[i]
                current_trade.hold_days = hold_days
                current_trade.exit_reason = exit_reason
                current_trade.exit_ts_ratio = ts_ratio if not pd.isna(ts_ratio) else 0
                current_trade.exit_ivrv = ivrv if not pd.isna(ivrv) else 0
                current_trade.exit_smh_price = df[idx_ticker].iloc[i]
                current_trade.cost_slippage += cfg.stock_slippage + cfg.options_slippage
                current_trade.pnl_total = (
                    current_trade.pnl_underlying
                    + current_trade.pnl_theta
                    + current_trade.pnl_gamma
                    + current_trade.pnl_vol_move
                    - current_trade.cost_slippage
                )
                self.trades.append(current_trade)
                daily_pnl -= (cfg.stock_slippage + cfg.options_slippage)
                current_trade = None
                position = "FLAT"
                hold_days = 0

            # --- ENTRY LOGIC (complacent side only) ---
            if position == "FLAT" and not pd.isna(ts_pctile):
                if regime == "COMPLACENT":
                    position = "SHORT_COMPLACENT"
                    hold_days = 1
                    trade_id += 1
                    current_trade = Trade(
                        trade_id=trade_id,
                        regime="COMPLACENT",
                        entry_date=df.index[i],
                        entry_ts_ratio=ts_ratio if not pd.isna(ts_ratio) else 0,
                        entry_ivrv=ivrv if not pd.isna(ivrv) else 0,
                        entry_corr=corr if not pd.isna(corr) else 0,
                        entry_skew=df["basket_skew"].iloc[i] if not pd.isna(df["basket_skew"].iloc[i]) else 0,
                        entry_smh_price=df[idx_ticker].iloc[i],
                    )
                    current_trade.cost_slippage += cfg.stock_slippage + cfg.options_slippage
                    daily_pnl -= (cfg.stock_slippage + cfg.options_slippage)

            # --- DAILY P&L ---
            if position != "FLAT" and current_trade is not None and hold_days > 0:
                idx_ret = df[f"{idx_ticker}_ret"].iloc[i]
                if pd.isna(idx_ret):
                    idx_ret = 0.0

                # Basket return (equal weight constituents)
                constituents = [t for t in cfg.tickers if t != cfg.index_ticker]
                basket_ret = np.nanmean([df[f"{t}_ret"].iloc[i] for t in constituents])
                if pd.isna(basket_ret):
                    basket_ret = 0.0

                # Vol changes
                rv_today = df[f"{idx_ticker}_rv_short"].iloc[i]
                rv_yesterday = df[f"{idx_ticker}_rv_short"].iloc[i-1] if i > 0 else rv_today
                d_rv = (rv_today - rv_yesterday) if not pd.isna(rv_today) and not pd.isna(rv_yesterday) else 0.0

                if position == "SHORT_COMPLACENT":
                    und_w = cfg.underlying_weight
                    opt_w = cfg.options_weight

                    # SHORT underlying (short SMH + constituents)
                    und_pnl = -und_w * (0.5 * idx_ret + 0.5 * basket_ret)

                    # BUY vol (long straddle): pay theta, earn gamma
                    theta_pnl = -opt_w * cfg.theta_daily_long
                    # Gamma benefit: profit from ALL big moves, not just exceeding implied
                    # Long straddle profits from absolute moves regardless of direction
                    realized_sq = basket_ret**2
                    expected_daily_var = (rv_today / np.sqrt(252))**2 if not pd.isna(rv_today) else 0.0
                    gamma_pnl = opt_w * 0.5 * max(0, realized_sq - expected_daily_var) * 100
                    # Vol move: long vol benefits from IV rising
                    vol_pnl = opt_w * 0.8 * d_rv

                else:
                    und_pnl = theta_pnl = gamma_pnl = vol_pnl = 0.0

                current_trade.pnl_underlying += und_pnl
                current_trade.pnl_theta += theta_pnl
                current_trade.pnl_gamma += gamma_pnl
                current_trade.pnl_vol_move += vol_pnl

                daily_pnl += und_pnl + theta_pnl + gamma_pnl + vol_pnl

            equity *= (1 + daily_pnl)
            self.daily_returns.append(daily_pnl)
            self.daily_positions.append(position)
            self.daily_equity.append(equity)
            self.dates.append(df.index[i])

        # Close any open trade
        if current_trade is not None:
            current_trade.exit_date = df.index[-1]
            current_trade.hold_days = hold_days
            current_trade.exit_reason = "end_of_data"
            current_trade.exit_smh_price = df[idx_ticker].iloc[-1]
            current_trade.pnl_total = (
                current_trade.pnl_underlying
                + current_trade.pnl_theta
                + current_trade.pnl_gamma
                + current_trade.pnl_vol_move
                - current_trade.cost_slippage
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
            "Profit Factor": f"{abs(sum(t.pnl_total for t in wins) / min(sum(t.pnl_total for t in losses), -0.0001)):.2f}" if losses else "inf",
            "Avg Hold Days": f"{np.mean([t.hold_days for t in self.trades]):.1f}" if self.trades else "0",
            "Days in Market": f"{sum(1 for p in self.daily_positions if p != 'FLAT')} / {len(self.daily_positions)}",
            "Trades/Year": f"{len(self.trades) / max(len(self.daily_positions) / 252, 1):.1f}",
        }

    def trade_log_df(self) -> pd.DataFrame:
        records = []
        for t in self.trades:
            records.append({
                "Trade #": t.trade_id,
                "Regime": t.regime,
                "Entry Date": t.entry_date,
                "Exit Date": t.exit_date,
                "Hold Days": t.hold_days,
                "Exit Reason": t.exit_reason,
                "Entry TS Ratio": round(t.entry_ts_ratio, 3),
                "Entry IV/RV": round(t.entry_ivrv, 3),
                "Entry Corr": round(t.entry_corr, 3),
                "Entry Skew": round(t.entry_skew, 3),
                "SMH Entry": f"${t.entry_smh_price:.2f}",
                "SMH Exit": f"${t.exit_smh_price:.2f}",
                "P&L Underlying": round(t.pnl_underlying, 5),
                "P&L Theta": round(t.pnl_theta, 5),
                "P&L Gamma": round(t.pnl_gamma, 5),
                "P&L Vol Move": round(t.pnl_vol_move, 5),
                "Cost Slippage": round(t.cost_slippage, 5),
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
        print("\n" + "=" * 75)
        print("  COMPLACENCY FADE STRATEGY — BACKTEST REPORT")
        print("  Short underlying + Long vol when complacency is extreme")
        print("=" * 75)

        print("\n--- Portfolio Metrics ---")
        for k, v in self.portfolio_metrics().items():
            print(f"  {k:.<35} {v}")

        print("\n--- P&L Attribution (all trades) ---")
        if self.trades:
            total_und = sum(t.pnl_underlying for t in self.trades)
            total_theta = sum(t.pnl_theta for t in self.trades)
            total_gamma = sum(t.pnl_gamma for t in self.trades)
            total_vol = sum(t.pnl_vol_move for t in self.trades)
            total_slip = sum(t.cost_slippage for t in self.trades)
            print(f"  Underlying (directional):   {total_und:+.4f}")
            print(f"  Theta (time decay):         {total_theta:+.4f}")
            print(f"  Gamma (realized moves):     {total_gamma:+.4f}")
            print(f"  Vol Move (IV changes):      {total_vol:+.4f}")
            print(f"  Slippage (entry+exit):      {-total_slip:+.4f}")
            print(f"  {'─' * 45}")
            net = total_und + total_theta + total_gamma + total_vol - total_slip
            print(f"  NET:                        {net:+.4f}")

        # Exit reason breakdown
        print("\n--- By Exit Reason ---")
        reasons = set(t.exit_reason for t in self.trades)
        for reason in sorted(reasons):
            trades = [t for t in self.trades if t.exit_reason == reason]
            if trades:
                wins = sum(1 for t in trades if t.pnl_total > 0)
                total_pnl = sum(t.pnl_total for t in trades)
                print(f"  {reason}: {len(trades)} trades | Win rate: {wins/len(trades):.0%} | "
                      f"Total P&L: {total_pnl:+.4f}")

        print("\n--- Trade Log ---")
        log_df = self.trade_log_df()
        if not log_df.empty:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 220)
            pd.set_option("display.max_colwidth", 12)
            print(log_df.to_string(index=False))
        else:
            print("  No trades executed.")

        print("\n" + "=" * 75)


# ============================================================
# MAIN
# ============================================================

def run_universe(universe_name, tickers, config):
    """Run the strategy on a single universe and return the backtester."""
    cfg = Config(
        tickers=tickers,
        index_ticker=tickers[0],
        backtest_years=config.backtest_years,
        complacent_composite_pctile=config.complacent_composite_pctile,
        underlying_weight=config.underlying_weight,
        options_weight=config.options_weight,
        min_hold_days=config.min_hold_days,
        max_hold_days=config.max_hold_days,
        signal_lookback=config.signal_lookback,
    )

    print(f"\n  [{universe_name}] Loading {', '.join(tickers)}...")
    try:
        df = load_data(cfg)
        df = compute_vol_features(df, cfg)
        df = classify_regimes(df, cfg)
        bt = VolRegimeBacktester(cfg)
        bt.run(df)

        complacent_days = (df["regime"] == "COMPLACENT").sum()
        print(f"  [{universe_name}] {len(df)} days | {complacent_days} complacent days | {len(bt.trades)} trades")
        return bt, df
    except Exception as e:
        print(f"  [{universe_name}] FAILED: {e}")
        return None, None


def main():
    config = Config()

    print("=" * 75)
    print("  COMPLACENCY FADE — MULTI-UNIVERSE BACKTESTER")
    print("  Signal: Short underlying + Long vol at extreme complacency")
    print("=" * 75)

    # Run primary universe
    all_backtesters = []
    universe_results = []

    primary_bt, primary_df = run_universe("SMH", config.tickers, config)
    if primary_bt:
        all_backtesters.append(("SMH", primary_bt))

    # Run additional universes
    for name, tickers in config.extra_universes:
        bt, df = run_universe(name, tickers, config)
        if bt:
            all_backtesters.append((name, bt))

    # Combined portfolio: equal-weight all universes
    print("\n" + "=" * 75)
    print("  INDIVIDUAL UNIVERSE RESULTS")
    print("=" * 75)

    all_trades = []
    combined_daily_rets = None

    for name, bt in all_backtesters:
        metrics = bt.portfolio_metrics()
        n_trades = len(bt.trades)
        sharpe = metrics["Sharpe Ratio"]
        total_ret = metrics["Total Return"]
        win_rate = metrics["Win Rate"]
        print(f"\n  {name:6s} | Trades: {n_trades:3d} | Sharpe: {sharpe} | "
              f"Return: {total_ret} | Win Rate: {win_rate}")

        # Collect trades with universe tag
        for t in bt.trades:
            t.regime = f"{name}_{t.regime}"
            all_trades.append(t)

        # Equal-weight daily returns
        rets = np.array(bt.daily_returns)
        if combined_daily_rets is None:
            combined_daily_rets = rets / len(all_backtesters)
        else:
            min_len = min(len(combined_daily_rets), len(rets))
            combined_daily_rets = combined_daily_rets[:min_len] + rets[:min_len] / len(all_backtesters)

    # Combined metrics
    if combined_daily_rets is not None and len(combined_daily_rets) > 0:
        equity = np.cumprod(1 + combined_daily_rets)
        ann_ret = np.mean(combined_daily_rets) * 252
        ann_vol = np.std(combined_daily_rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dd = np.min(dd)

        wins = [t for t in all_trades if t.pnl_total > 0]
        losses = [t for t in all_trades if t.pnl_total <= 0]

        print("\n" + "=" * 75)
        print("  COMBINED PORTFOLIO (equal-weight across universes)")
        print("=" * 75)
        print(f"\n  Total Return:          {equity[-1] - 1:.2%}")
        print(f"  Annualized Return:     {ann_ret:.2%}")
        print(f"  Annualized Volatility: {ann_vol:.2%}")
        print(f"  Sharpe Ratio:          {sharpe:.2f}")
        print(f"  Max Drawdown:          {max_dd:.2%}")
        print(f"  Total Trades:          {len(all_trades)}")
        print(f"  Win Rate:              {len(wins) / max(len(all_trades), 1):.1%}")
        print(f"  Trades/Year:           {len(all_trades) / max(len(combined_daily_rets) / 252, 1):.1f}")

        # P&L attribution combined
        total_und = sum(t.pnl_underlying for t in all_trades)
        total_theta = sum(t.pnl_theta for t in all_trades)
        total_gamma = sum(t.pnl_gamma for t in all_trades)
        total_vol = sum(t.pnl_vol_move for t in all_trades)
        total_slip = sum(t.cost_slippage for t in all_trades)
        print(f"\n  --- P&L Attribution ---")
        print(f"  Underlying:   {total_und:+.4f}")
        print(f"  Theta:        {total_theta:+.4f}")
        print(f"  Gamma:        {total_gamma:+.4f}")
        print(f"  Vol Move:     {total_vol:+.4f}")
        print(f"  Slippage:     {-total_slip:+.4f}")
        net = total_und + total_theta + total_gamma + total_vol - total_slip
        print(f"  NET:          {net:+.4f}")

    # Save combined trade log
    records = []
    for t in all_trades:
        records.append({
            "Universe": t.regime.split("_")[0],
            "Trade #": t.trade_id,
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Hold Days": t.hold_days,
            "Exit Reason": t.exit_reason,
            "Entry TS": round(t.entry_ts_ratio, 3),
            "Entry IV/RV": round(t.entry_ivrv, 3),
            "Entry Corr": round(t.entry_corr, 3),
            "P&L Underlying": round(t.pnl_underlying, 5),
            "P&L Gamma": round(t.pnl_gamma, 5),
            "P&L Vol": round(t.pnl_vol_move, 5),
            "Total P&L": round(t.pnl_total, 5),
        })

    combined_log = pd.DataFrame(records)
    log_path = os.path.join(OUTPUT_DIR, "regime_trade_log.csv")
    combined_log.to_csv(log_path, index=False)
    print(f"\n  Combined trade log saved to: {log_path}")

    # Current signals
    print("\n  --- Current Signals ---")
    if primary_df is not None:
        latest = primary_df.iloc[-1]
        print(f"  SMH regime: {latest['regime']} | Composite pctile: {latest['composite_pctile']:.0f}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
