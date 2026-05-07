"""
Complacency Fade Strategy
=========================

Implementation driven by STRATEGY.md.

Core idea:
- Identify extreme complacency in AI semiconductor volatility conditions.
- Enter long vol trades only in that regime.
- Exit when the signal normalizes, flips to fear, or the max hold is reached.
"""

import os
import sys
from dataclasses import dataclass
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "backtest_output")
YF_CACHE_DIR = os.path.join(BASE_DIR, ".yfinance_cache_spiny")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YF_CACHE_DIR, exist_ok=True)

try:
    yf.set_tz_cache_location(YF_CACHE_DIR)
except Exception:
    pass


@dataclass
class Config:
    backtest_years: int = 5
    tickers: tuple[str, ...] = ("SMH", "NVDA", "AVGO", "TSM")
    index_ticker: str = "SMH"
    extra_universes: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("SOXX", ("SOXX", "NVDA", "AVGO", "TSM")),
    )

    rv_short_window: int = 10
    rv_long_window: int = 63
    corr_window: int = 10
    skew_window: int = 21
    signal_lookback: int = 252

    fear_pctile: float = 75.0
    complacent_pctile: float = 15.0
    normalize_exit_pctile: float = 50.0

    underlying_weight: float = 0.5
    options_weight: float = 0.5
    enable_underlying_leg: bool = False
    enable_options_leg: bool = True

    min_hold_days: int = 7
    max_hold_days: int = 42

    stock_slippage: float = 0.001
    options_slippage: float = 0.012
    theta_daily_long: float = 0.00035


@dataclass
class Trade:
    trade_id: int
    universe: str
    entry_date: object = None
    exit_date: object = None
    hold_days: int = 0
    exit_reason: str = ""

    entry_composite_pctile: float = 0.0
    entry_ts_ratio: float = 0.0
    entry_ivrv: float = 0.0
    entry_corr: float = 0.0
    entry_skew: float = 0.0
    entry_index_price: float = 0.0

    exit_composite_pctile: float = 0.0
    exit_index_price: float = 0.0

    pnl_underlying: float = 0.0
    pnl_theta: float = 0.0
    pnl_gamma: float = 0.0
    pnl_vol_move: float = 0.0
    cost_slippage: float = 0.0
    pnl_total: float = 0.0


def load_data(config: Config) -> pd.DataFrame:
    period = f"{config.backtest_years}y"
    frames: dict[str, pd.Series] = {}

    for ticker in config.tickers:
        print(f"  Downloading {ticker}...")
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            raise RuntimeError(f"Failed to download {ticker}")
        frames[ticker] = data["Close"].squeeze()

    print("  Downloading ^VXN...")
    vxn = yf.download("^VXN", period=period, interval="1d", auto_adjust=True, progress=False)
    if not vxn.empty:
        frames["VXN"] = vxn["Close"].squeeze()

    df = pd.DataFrame(frames).dropna()
    if df.empty:
        raise RuntimeError("No aligned price history available after download.")

    print(f"  Combined dataset: {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def compute_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    constituents = [ticker for ticker in config.tickers if ticker != config.index_ticker]

    for ticker in config.tickers:
        ret_col = f"{ticker}_ret"
        rv_short_col = f"{ticker}_rv_short"
        rv_long_col = f"{ticker}_rv_long"
        skew_col = f"{ticker}_skew"

        df[ret_col] = np.log(df[ticker]).diff()
        df[rv_short_col] = df[ret_col].rolling(config.rv_short_window).std() * np.sqrt(252)
        df[rv_long_col] = df[ret_col].rolling(config.rv_long_window).std() * np.sqrt(252)
        df[f"{ticker}_ts_ratio"] = df[rv_short_col] / df[rv_long_col].replace(0, np.nan)
        df[skew_col] = df[ret_col].rolling(config.skew_window).skew()

    if "VXN" in df.columns:
        df["base_iv_proxy"] = df["VXN"] / 100.0
    else:
        df["base_iv_proxy"] = df[f"{config.index_ticker}_rv_short"] * 1.15

    for ticker in constituents:
        vol_ratio = df[f"{ticker}_rv_short"] / df[f"{config.index_ticker}_rv_short"].replace(0, np.nan)
        vol_ratio = vol_ratio.ewm(halflife=21, adjust=False).mean().shift(1).fillna(1.5)
        df[f"{ticker}_iv_proxy"] = df["base_iv_proxy"] * vol_ratio
        df[f"{ticker}_ivrv"] = (
            df[f"{ticker}_iv_proxy"] / df[f"{ticker}_rv_short"].replace(0, np.nan)
        ).clip(0.3, 5.0)

    # Approximate index IV/RV so the report remains self-contained.
    df[f"{config.index_ticker}_iv_proxy"] = df["base_iv_proxy"]
    df[f"{config.index_ticker}_ivrv"] = (
        df[f"{config.index_ticker}_iv_proxy"] / df[f"{config.index_ticker}_rv_short"].replace(0, np.nan)
    ).clip(0.3, 5.0)

    corr_cols = []
    for i, left in enumerate(constituents):
        for right in constituents[i + 1 :]:
            corr_col = f"corr_{left}_{right}"
            df[corr_col] = df[f"{left}_ret"].rolling(config.corr_window).corr(df[f"{right}_ret"])
            corr_cols.append(corr_col)

    df["avg_constituent_corr"] = df[corr_cols].mean(axis=1) if corr_cols else np.nan
    df["basket_ts_ratio"] = df[[f"{ticker}_ts_ratio" for ticker in constituents]].mean(axis=1)
    df["basket_ivrv"] = df[[f"{ticker}_ivrv" for ticker in constituents]].mean(axis=1)
    df["basket_skew"] = df[[f"{ticker}_skew" for ticker in constituents]].mean(axis=1)

    return df


def classify_regimes(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    lookback = config.signal_lookback

    # The markdown defines a complacency score: lower values mean more complacent.
    df["ts_pctile"] = df["basket_ts_ratio"].rolling(lookback).rank(pct=True) * 100
    df["ivrv_pctile"] = df["basket_ivrv"].rolling(lookback).rank(pct=True) * 100
    df["corr_pctile"] = df["avg_constituent_corr"].rolling(lookback).rank(pct=True) * 100
    df["skew_pctile"] = df["basket_skew"].rolling(lookback).rank(pct=True) * 100

    df["composite_score"] = (
        0.35 * df["ts_pctile"]
        + 0.30 * df["ivrv_pctile"]
        + 0.20 * df["corr_pctile"]
        + 0.15 * df["skew_pctile"]
    )
    df["composite_pctile"] = df["composite_score"].rolling(lookback).rank(pct=True) * 100

    regimes: list[str] = []
    for value in df["composite_pctile"]:
        if pd.isna(value):
            regimes.append("NEUTRAL")
        elif value <= config.complacent_pctile:
            regimes.append("COMPLACENT")
        elif value >= config.fear_pctile:
            regimes.append("FEAR")
        else:
            regimes.append("NEUTRAL")

    df["regime"] = regimes
    return df


class ComplacencyFadeBacktester:
    def __init__(self, config: Config, universe_name: str):
        self.config = config
        self.universe_name = universe_name
        self.trades: list[Trade] = []
        self.daily_returns: list[float] = []
        self.daily_positions: list[str] = []
        self.daily_equity: list[float] = []
        self.dates: list[object] = []

    def run(self, df: pd.DataFrame) -> "ComplacencyFadeBacktester":
        cfg = self.config
        idx = cfg.index_ticker
        constituents = [ticker for ticker in cfg.tickers if ticker != idx]

        position = "FLAT"
        hold_days = 0
        trade_id = 0
        equity = 1.0
        current_trade: Optional[Trade] = None

        for i in range(1, len(df)):
            prior_regime = df["regime"].iloc[i - 1]
            prior_composite = df["composite_pctile"].iloc[i - 1]
            daily_pnl = 0.0

            if position != "FLAT":
                hold_days += 1
                exit_reason = ""

                if hold_days >= cfg.max_hold_days:
                    exit_reason = "max_hold"
                elif hold_days >= cfg.min_hold_days and prior_regime == "FEAR":
                    exit_reason = "regime_flip_fear"
                elif hold_days >= cfg.min_hold_days and not pd.isna(prior_composite):
                    if prior_composite > cfg.normalize_exit_pctile:
                        exit_reason = "signal_normalized"

                if exit_reason and current_trade is not None:
                    current_trade.exit_date = df.index[i]
                    current_trade.hold_days = hold_days
                    current_trade.exit_reason = exit_reason
                    current_trade.exit_composite_pctile = float(prior_composite) if not pd.isna(prior_composite) else 0.0
                    current_trade.exit_index_price = float(df[idx].iloc[i])
                    current_trade.cost_slippage += cfg.stock_slippage + cfg.options_slippage
                    current_trade.pnl_total = (
                        current_trade.pnl_underlying
                        + current_trade.pnl_theta
                        + current_trade.pnl_gamma
                        + current_trade.pnl_vol_move
                        - current_trade.cost_slippage
                    )
                    self.trades.append(current_trade)
                    daily_pnl -= cfg.stock_slippage + cfg.options_slippage
                    current_trade = None
                    position = "FLAT"
                    hold_days = 0

            if position == "FLAT" and prior_regime == "COMPLACENT":
                trade_id += 1
                position = "SHORT_UNDERLYING_LONG_VOL"
                hold_days = 1
                current_trade = Trade(
                    trade_id=trade_id,
                    universe=self.universe_name,
                    entry_date=df.index[i],
                    entry_composite_pctile=float(prior_composite) if not pd.isna(prior_composite) else 0.0,
                    entry_ts_ratio=float(df["basket_ts_ratio"].iloc[i - 1]) if not pd.isna(df["basket_ts_ratio"].iloc[i - 1]) else 0.0,
                    entry_ivrv=float(df["basket_ivrv"].iloc[i - 1]) if not pd.isna(df["basket_ivrv"].iloc[i - 1]) else 0.0,
                    entry_corr=float(df["avg_constituent_corr"].iloc[i - 1]) if not pd.isna(df["avg_constituent_corr"].iloc[i - 1]) else 0.0,
                    entry_skew=float(df["basket_skew"].iloc[i - 1]) if not pd.isna(df["basket_skew"].iloc[i - 1]) else 0.0,
                    entry_index_price=float(df[idx].iloc[i]),
                )
                current_trade.cost_slippage += cfg.stock_slippage + cfg.options_slippage
                daily_pnl -= cfg.stock_slippage + cfg.options_slippage

            if position != "FLAT" and current_trade is not None:
                idx_ret = df[f"{idx}_ret"].iloc[i]
                idx_ret = 0.0 if pd.isna(idx_ret) else float(idx_ret)

                basket_ret = float(np.nanmean([df[f"{ticker}_ret"].iloc[i] for ticker in constituents]))
                if pd.isna(basket_ret):
                    basket_ret = 0.0

                rv_today = df[f"{idx}_rv_short"].iloc[i]
                rv_prev = df[f"{idx}_rv_short"].iloc[i - 1]
                rv_today = 0.0 if pd.isna(rv_today) else float(rv_today)
                rv_prev = rv_today if pd.isna(rv_prev) else float(rv_prev)
                d_rv = rv_today - rv_prev

                und_pnl = -cfg.underlying_weight * (0.5 * idx_ret + 0.5 * basket_ret)
                if not cfg.enable_underlying_leg:
                    und_pnl = 0.0

                opt_w = cfg.options_weight if cfg.enable_options_leg else 0.0
                theta_pnl = -opt_w * cfg.theta_daily_long

                realized_sq = basket_ret**2
                implied_daily_var = (rv_today / np.sqrt(252)) ** 2
                gamma_pnl = opt_w * 0.5 * max(0.0, realized_sq - implied_daily_var) * 100.0
                vol_move_pnl = opt_w * 0.8 * d_rv

                current_trade.pnl_underlying += und_pnl
                current_trade.pnl_theta += theta_pnl
                current_trade.pnl_gamma += gamma_pnl
                current_trade.pnl_vol_move += vol_move_pnl

                daily_pnl += und_pnl + theta_pnl + gamma_pnl + vol_move_pnl

            equity *= 1.0 + daily_pnl
            self.daily_returns.append(daily_pnl)
            self.daily_positions.append(position)
            self.daily_equity.append(equity)
            self.dates.append(df.index[i])

        if current_trade is not None:
            current_trade.exit_date = df.index[-1]
            current_trade.hold_days = hold_days
            current_trade.exit_reason = "end_of_data"
            current_trade.exit_composite_pctile = (
                float(df["composite_pctile"].iloc[-1]) if not pd.isna(df["composite_pctile"].iloc[-1]) else 0.0
            )
            current_trade.exit_index_price = float(df[idx].iloc[-1])
            current_trade.pnl_total = (
                current_trade.pnl_underlying
                + current_trade.pnl_theta
                + current_trade.pnl_gamma
                + current_trade.pnl_vol_move
                - current_trade.cost_slippage
            )
            self.trades.append(current_trade)

        return self

    def portfolio_metrics(self) -> dict[str, str]:
        if not self.daily_equity:
            return {
                "Total Return": "0.00%",
                "Annualized Return": "0.00%",
                "Annualized Volatility": "0.00%",
                "Sharpe Ratio": "0.00",
                "Max Drawdown": "0.00%",
                "Total Trades": "0",
                "Win Rate": "0.0%",
            }

        rets = np.array(self.daily_returns, dtype=float)
        equity = np.array(self.daily_equity, dtype=float)
        ann_ret = np.mean(rets) * 252
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        wins = [trade for trade in self.trades if trade.pnl_total > 0]
        return {
            "Total Return": f"{equity[-1] - 1:.2%}",
            "Annualized Return": f"{ann_ret:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{drawdown.min():.2%}",
            "Total Trades": str(len(self.trades)),
            "Win Rate": f"{len(wins) / max(len(self.trades), 1):.1%}",
        }

    def equity_curve_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": self.dates,
                "Equity": self.daily_equity,
                "Daily Return": self.daily_returns,
                "Position": self.daily_positions,
            }
        )

    def trade_log_df(self) -> pd.DataFrame:
        rows = []
        for trade in self.trades:
            rows.append(
                {
                    "Universe": trade.universe,
                    "Trade #": trade.trade_id,
                    "Entry Date": trade.entry_date,
                    "Exit Date": trade.exit_date,
                    "Hold Days": trade.hold_days,
                    "Exit Reason": trade.exit_reason,
                    "Entry Composite %ile": round(trade.entry_composite_pctile, 2),
                    "Entry TS Ratio": round(trade.entry_ts_ratio, 3),
                    "Entry IV/RV": round(trade.entry_ivrv, 3),
                    "Entry Corr": round(trade.entry_corr, 3),
                    "Entry Skew": round(trade.entry_skew, 3),
                    "Entry Index Px": round(trade.entry_index_price, 2),
                    "Exit Composite %ile": round(trade.exit_composite_pctile, 2),
                    "Exit Index Px": round(trade.exit_index_price, 2),
                    "P&L Underlying": round(trade.pnl_underlying, 5),
                    "P&L Theta": round(trade.pnl_theta, 5),
                    "P&L Gamma": round(trade.pnl_gamma, 5),
                    "P&L Vol Move": round(trade.pnl_vol_move, 5),
                    "Cost Slippage": round(trade.cost_slippage, 5),
                    "Total P&L": round(trade.pnl_total, 5),
                }
            )
        return pd.DataFrame(rows)


def run_universe(universe_name: str, tickers: tuple[str, ...], root_config: Config) -> tuple[Optional[ComplacencyFadeBacktester], Optional[pd.DataFrame]]:
    config = Config(
        backtest_years=root_config.backtest_years,
        tickers=tickers,
        index_ticker=tickers[0],
        extra_universes=(),
        rv_short_window=root_config.rv_short_window,
        rv_long_window=root_config.rv_long_window,
        corr_window=root_config.corr_window,
        skew_window=root_config.skew_window,
        signal_lookback=root_config.signal_lookback,
        fear_pctile=root_config.fear_pctile,
        complacent_pctile=root_config.complacent_pctile,
        normalize_exit_pctile=root_config.normalize_exit_pctile,
        underlying_weight=root_config.underlying_weight,
        options_weight=root_config.options_weight,
        enable_underlying_leg=root_config.enable_underlying_leg,
        enable_options_leg=root_config.enable_options_leg,
        min_hold_days=root_config.min_hold_days,
        max_hold_days=root_config.max_hold_days,
        stock_slippage=root_config.stock_slippage,
        options_slippage=root_config.options_slippage,
        theta_daily_long=root_config.theta_daily_long,
    )

    print(f"\n[{universe_name}] Loading {', '.join(tickers)}...")
    try:
        df = load_data(config)
        df = compute_features(df, config)
        df = classify_regimes(df, config)
        bt = ComplacencyFadeBacktester(config, universe_name)
        bt.run(df)
        complacent_days = int((df["regime"] == "COMPLACENT").sum())
        print(f"[{universe_name}] {len(df)} days | {complacent_days} complacent days | {len(bt.trades)} trades")
        return bt, df
    except Exception as exc:
        print(f"[{universe_name}] FAILED: {exc}")
        return None, None


def save_outputs(backtesters: list[ComplacencyFadeBacktester]) -> None:
    trade_logs = []
    equity_curves = []

    for bt in backtesters:
        trade_log = bt.trade_log_df()
        if not trade_log.empty:
            trade_logs.append(trade_log)

        curve = bt.equity_curve_df().copy()
        if not curve.empty:
            curve.insert(0, "Universe", bt.universe_name)
            equity_curves.append(curve)

    if trade_logs:
        trades_path = os.path.join(OUTPUT_DIR, "spiny_trade_log.csv")
        pd.concat(trade_logs, ignore_index=True).to_csv(trades_path, index=False)
        print(f"\nSaved trade log to: {trades_path}")

    if equity_curves:
        curve_path = os.path.join(OUTPUT_DIR, "spiny_equity_curve.csv")
        pd.concat(equity_curves, ignore_index=True).to_csv(curve_path, index=False)
        print(f"Saved equity curves to: {curve_path}")


def print_summary(backtesters: list[ComplacencyFadeBacktester], primary_df: Optional[pd.DataFrame]) -> None:
    print("\n" + "=" * 75)
    print("COMPLACENCY FADE SUMMARY")
    print("=" * 75)

    for bt in backtesters:
        metrics = bt.portfolio_metrics()
        print(
            f"\n{bt.universe_name:6s} | Trades: {metrics['Total Trades']:>3s} | "
            f"Sharpe: {metrics['Sharpe Ratio']} | Return: {metrics['Total Return']} | "
            f"Win Rate: {metrics['Win Rate']}"
        )

    if primary_df is not None and not primary_df.empty:
        latest = primary_df.iloc[-1]
        print("\nCurrent primary-universe signal:")
        print(
            f"Regime={latest['regime']} | Composite %ile={latest['composite_pctile']:.0f} | "
            f"TS={latest['basket_ts_ratio']:.2f} | Corr={latest['avg_constituent_corr']:.2f}"
        )

    print("\n" + "=" * 75)


def main() -> None:
    root_config = Config()
    backtesters: list[ComplacencyFadeBacktester] = []
    leg_mode = (
        "combined"
        if root_config.enable_underlying_leg and root_config.enable_options_leg
        else "short-only"
        if root_config.enable_underlying_leg
        else "options-only"
        if root_config.enable_options_leg
        else "disabled"
    )

    print("=" * 75)
    print("COMPLACENCY FADE - STRATEGY.MD IMPLEMENTATION")
    print(f"Signal mode: {leg_mode}")
    print("=" * 75)

    primary_bt, primary_df = run_universe("SMH", root_config.tickers, root_config)
    if primary_bt is not None:
        backtesters.append(primary_bt)

    for universe_name, tickers in root_config.extra_universes:
        bt, _ = run_universe(universe_name, tickers, root_config)
        if bt is not None:
            backtesters.append(bt)

    save_outputs(backtesters)
    print_summary(backtesters, primary_df)


if __name__ == "__main__":
    main()
