"""
Monte Carlo Simulation Backtester
=================================
Tests whether the Complacency Fade strategy's edge is real or an artifact
of the specific historical path we backtested on.

Approach:
---------
1. Estimate statistical properties from real data (vol levels, correlations,
   vol-of-vol, mean-reversion speeds, fat tails, correlation clustering)
2. Generate 1000+ synthetic paths that share these properties but with
   different random draws
3. Run the strategy on each path
4. Analyze distribution of outcomes: Sharpe, returns, drawdowns
5. Compare to random entry/exit as a null hypothesis

Key questions answered:
- What % of simulations are profitable?
- Is the Sharpe significantly different from zero?
- How much of the backtest result is luck vs edge?
- What's the worst-case scenario?
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

YF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yfinance_cache_mc")
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
class MCConfig:
    # Simulation
    n_simulations: int = 1000
    n_days: int = 1255            # ~5 years
    n_assets: int = 4             # index + 3 constituents

    # Strategy (same as vol_regime_backtester.py)
    rv_short_window: int = 10
    rv_long_window: int = 63
    skew_window: int = 21
    signal_lookback: int = 252
    complacent_pctile: float = 15
    min_hold_days: int = 7
    max_hold_days: int = 42
    underlying_weight: float = 0.5
    options_weight: float = 0.5
    stock_slippage: float = 0.001
    options_slippage: float = 0.012
    theta_daily_long: float = 0.00035

    # Random baseline
    n_random_strategies: int = 1000


# ============================================================
# ESTIMATE PARAMETERS FROM REAL DATA
# ============================================================

@dataclass
class MarketParams:
    # Marginal return distributions
    daily_means: np.ndarray = None       # (4,) annualized drift
    daily_vols: np.ndarray = None        # (4,) annualized vol
    correlation_matrix: np.ndarray = None  # (4,4) average correlation
    vol_of_vol: np.ndarray = None        # (4,) how much vol itself fluctuates
    mean_reversion_speed: float = 0.05   # OU speed for vol process
    tail_df: float = 5.0                 # Student-t degrees of freedom (fat tails)
    corr_clustering: float = 0.85        # autocorrelation of correlation regimes


def estimate_params_from_real_data() -> MarketParams:
    """Download real data and extract statistical properties for simulation."""
    print("  Estimating market parameters from real data...")
    tickers = ["SMH", "NVDA", "AVGO", "TSM"]
    period = "5y"

    frames = {}
    for ticker in tickers:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if not data.empty:
            frames[ticker] = data["Close"].squeeze()

    df = pd.DataFrame(frames).dropna()
    log_rets = np.log(df).diff().dropna()

    params = MarketParams()
    params.daily_means = log_rets.mean().values * 252
    params.daily_vols = log_rets.std().values * np.sqrt(252)
    params.correlation_matrix = log_rets.corr().values

    # Vol-of-vol: std of rolling 21-day realized vol
    rolling_vols = log_rets.rolling(21).std() * np.sqrt(252)
    params.vol_of_vol = rolling_vols.std().values

    # Mean reversion speed: from OU fit on rolling vol
    for i, ticker in enumerate(tickers):
        rv = rolling_vols.iloc[:, i].dropna()
        rv_lag = rv.shift(1).dropna()
        rv_aligned = rv.iloc[1:]
        if len(rv_aligned) > 50:
            slope = np.polyfit(rv_lag.values, rv_aligned.values, 1)[0]
            params.mean_reversion_speed = max(1 - slope, 0.01)
            break

    # Fat tails: fit Student-t to pooled returns
    pooled = log_rets.values.flatten()
    pooled = pooled[~np.isnan(pooled)]
    try:
        df_t, _, _ = stats.t.fit(pooled / pooled.std())
        params.tail_df = max(min(df_t, 30), 3)
    except Exception:
        params.tail_df = 5.0

    # Correlation clustering
    rolling_corr = log_rets.iloc[:, 0].rolling(21).corr(log_rets.iloc[:, 1]).dropna()
    if len(rolling_corr) > 10:
        params.corr_clustering = max(rolling_corr.autocorr(lag=1), 0.5)

    print(f"    Daily vols (ann): {[f'{v:.1%}' for v in params.daily_vols]}")
    print(f"    Avg correlation: {params.correlation_matrix[0, 1:].mean():.2f}")
    print(f"    Vol-of-vol: {[f'{v:.1%}' for v in params.vol_of_vol]}")
    print(f"    Fat-tail df: {params.tail_df:.1f}")
    print(f"    Corr clustering: {params.corr_clustering:.2f}")

    return params


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_path(params: MarketParams, n_days: int, seed: int) -> pd.DataFrame:
    """
    Generate one synthetic market path with realistic properties:
    - Time-varying volatility (stochastic vol via OU process)
    - Fat-tailed returns (Student-t innovations)
    - Correlation clustering (vol-dependent correlation)
    - Volatility mean-reversion
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params.daily_vols)
    tickers = ["IDX", "A", "B", "C"]

    # Generate time-varying volatility via OU process
    daily_base_vol = params.daily_vols / np.sqrt(252)
    vols = np.zeros((n_days, n_assets))
    vols[0] = daily_base_vol

    mr_speed = params.mean_reversion_speed
    vov = params.vol_of_vol / np.sqrt(252) * 0.3  # scale vol-of-vol shocks

    for t in range(1, n_days):
        vol_shock = rng.normal(0, 1, n_assets) * vov
        vols[t] = vols[t-1] + mr_speed * (daily_base_vol - vols[t-1]) + vol_shock
        vols[t] = np.clip(vols[t], daily_base_vol * 0.3, daily_base_vol * 3.0)

    # Generate correlated fat-tailed returns
    # Use time-varying correlation (higher when vol is high)
    returns = np.zeros((n_days, n_assets))
    base_corr = params.correlation_matrix.copy()

    for t in range(n_days):
        # Correlation increases when vol is elevated (correlation clustering)
        vol_ratio = np.mean(vols[t] / daily_base_vol)
        corr_boost = np.clip((vol_ratio - 1.0) * 0.3, 0, 0.3)

        # Adjust correlation matrix
        corr_t = base_corr.copy()
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    corr_t[i, j] = min(corr_t[i, j] + corr_boost, 0.98)

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(corr_t)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_t = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(corr_t, 1.0)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr_t)
        except np.linalg.LinAlgError:
            L = np.eye(n_assets)

        # Student-t innovations
        z = rng.standard_t(df=params.tail_df, size=n_assets)
        z = z / np.sqrt(params.tail_df / (params.tail_df - 2))  # normalize variance
        correlated_z = L @ z

        returns[t] = vols[t] * correlated_z

    # Add small drift
    daily_drift = params.daily_means / 252
    returns += daily_drift[np.newaxis, :]

    # Build price series
    prices = np.exp(np.cumsum(returns, axis=0)) * 100  # start at 100

    # Build IV proxy (VXN-like): realized vol * random premium
    iv_premium = 1.0 + 0.15 * rng.standard_normal(n_days)
    iv_premium = np.clip(np.cumsum(0.02 * (1.12 - iv_premium)) + iv_premium, 0.8, 1.8)
    idx_rv = pd.Series(returns[:, 0]).rolling(21).std().values * np.sqrt(252)
    vxn_proxy = np.where(np.isnan(idx_rv), 0.20, idx_rv * iv_premium)
    vxn_proxy = np.clip(vxn_proxy, 0.08, 1.0)

    df = pd.DataFrame({
        tickers[0]: prices[:, 0],
        tickers[1]: prices[:, 1],
        tickers[2]: prices[:, 2],
        tickers[3]: prices[:, 3],
        "VXN": vxn_proxy * 100,
    }, index=pd.date_range("2021-01-01", periods=n_days, freq="B"))

    return df


# ============================================================
# STRATEGY LOGIC (simplified from vol_regime_backtester.py)
# ============================================================

def run_strategy_on_path(df: pd.DataFrame, config: MCConfig) -> dict:
    """Run the complacency fade strategy on a synthetic path. Returns metrics."""
    tickers = [c for c in df.columns if c != "VXN"]
    idx = tickers[0]
    constituents = tickers[1:]
    n = len(df)

    # Compute returns
    for t in tickers:
        df[f"{t}_ret"] = np.log(df[t]).diff()

    # Realized vols
    for t in tickers:
        df[f"{t}_rv_short"] = df[f"{t}_ret"].rolling(config.rv_short_window).std() * np.sqrt(252)
        df[f"{t}_rv_long"] = df[f"{t}_ret"].rolling(config.rv_long_window).std() * np.sqrt(252)

    # Term structure ratio
    ts_cols = []
    for t in constituents:
        col = f"{t}_ts"
        df[col] = df[f"{t}_rv_short"] / df[f"{t}_rv_long"].replace(0, np.nan)
        ts_cols.append(col)
    df["basket_ts"] = df[ts_cols].mean(axis=1)

    # IV-RV spread
    df[f"{idx}_iv_proxy"] = df["VXN"] / 100.0
    ivrv_cols = []
    for t in constituents:
        vol_ratio = (df[f"{t}_rv_short"] / df[f"{idx}_rv_short"].replace(0, np.nan)).ewm(halflife=21).mean().shift(1).fillna(1.5)
        df[f"{t}_iv"] = df[f"{idx}_iv_proxy"] * vol_ratio
        col = f"{t}_ivrv"
        df[col] = (df[f"{t}_iv"] / df[f"{t}_rv_short"].replace(0, np.nan)).clip(0.3, 5.0)
        ivrv_cols.append(col)
    df["basket_ivrv"] = df[ivrv_cols].mean(axis=1)

    # Correlation
    corr_cols = []
    for i, t1 in enumerate(constituents):
        for t2 in constituents[i+1:]:
            col = f"corr_{t1}_{t2}"
            df[col] = df[f"{t1}_ret"].rolling(config.rv_short_window).corr(df[f"{t2}_ret"])
            corr_cols.append(col)
    df["avg_corr"] = df[corr_cols].mean(axis=1) if corr_cols else 0.5

    # Skew
    skew_cols = []
    for t in constituents:
        col = f"{t}_skew"
        df[col] = df[f"{t}_ret"].rolling(config.skew_window).skew()
        skew_cols.append(col)
    df["basket_skew"] = df[skew_cols].mean(axis=1)

    # Composite score
    lookback = config.signal_lookback
    df["ts_pctile"] = df["basket_ts"].rolling(lookback).rank(pct=True) * 100
    df["ivrv_pctile"] = df["basket_ivrv"].rolling(lookback).rank(pct=True) * 100
    df["corr_pctile"] = df["avg_corr"].rolling(lookback).rank(pct=True) * 100
    df["skew_pctile"] = (-df["basket_skew"]).rolling(lookback).rank(pct=True) * 100

    df["composite"] = (0.35 * df["ts_pctile"] + 0.30 * df["ivrv_pctile"] +
                       0.20 * df["corr_pctile"] + 0.15 * df["skew_pctile"])
    df["composite_pctile"] = df["composite"].rolling(lookback).rank(pct=True) * 100

    # Classify regime
    regimes = np.where(df["composite_pctile"] <= config.complacent_pctile, "C",
                       np.where(df["composite_pctile"] >= 75, "F", "N"))

    # Run strategy
    position = 0  # 0=flat, -1=short
    hold_days = 0
    equity = 1.0
    daily_rets = []
    n_trades = 0
    trade_pnls = []
    current_pnl = 0.0

    idx_ret = df[f"{idx}_ret"].values
    basket_ret_vals = df[[f"{t}_ret" for t in constituents]].mean(axis=1).values
    rv_short = df[f"{idx}_rv_short"].values
    comp_pctile = df["composite_pctile"].values

    for i in range(1, n):
        dpnl = 0.0
        regime = regimes[i - 1]

        # Exit
        if position == -1:
            hold_days += 1
            should_exit = False
            if hold_days >= config.max_hold_days:
                should_exit = True
            elif hold_days >= config.min_hold_days and regime != "C":
                should_exit = True
            elif hold_days >= config.min_hold_days:
                cp = comp_pctile[i]
                if not np.isnan(cp) and cp > 50:
                    should_exit = True

            if should_exit:
                dpnl -= (config.stock_slippage + config.options_slippage)
                trade_pnls.append(current_pnl + dpnl)
                position = 0
                hold_days = 0
                current_pnl = 0.0

        # Entry
        if position == 0 and regime == "C":
            cp = comp_pctile[i - 1]
            if not np.isnan(cp) and cp <= config.complacent_pctile:
                position = -1
                hold_days = 1
                n_trades += 1
                entry_cost = config.stock_slippage + config.options_slippage
                dpnl -= entry_cost
                current_pnl = -entry_cost

        # Daily P&L if in position
        if position == -1 and hold_days > 0:
            ir = idx_ret[i] if not np.isnan(idx_ret[i]) else 0.0
            br = basket_ret_vals[i] if not np.isnan(basket_ret_vals[i]) else 0.0
            rv = rv_short[i] if not np.isnan(rv_short[i]) else 0.2

            und_pnl = -config.underlying_weight * (0.5 * ir + 0.5 * br)
            theta_pnl = -config.options_weight * config.theta_daily_long
            expected_var = (rv / np.sqrt(252))**2
            realized_sq = br**2
            gamma_pnl = config.options_weight * 0.5 * max(0, realized_sq - expected_var) * 100

            rv_prev = rv_short[i-1] if not np.isnan(rv_short[i-1]) else rv
            vol_pnl = config.options_weight * 0.8 * (rv - rv_prev)

            day_pnl = und_pnl + theta_pnl + gamma_pnl + vol_pnl
            dpnl += day_pnl
            current_pnl += day_pnl

        equity *= (1 + dpnl)
        daily_rets.append(dpnl)

    # Close open trade
    if position == -1:
        trade_pnls.append(current_pnl)

    rets = np.array(daily_rets)
    ann_ret = np.mean(rets) * 252
    ann_vol = np.std(rets) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    eq = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd = np.min(dd) if len(dd) > 0 else 0.0

    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = wins / max(len(trade_pnls), 1)

    return {
        "total_return": equity - 1,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": abs(sum(p for p in trade_pnls if p > 0) / min(sum(p for p in trade_pnls if p <= 0), -1e-10)),
    }


def run_random_strategy(df: pd.DataFrame, config: MCConfig, seed: int) -> dict:
    """Random entry/exit baseline with same trade frequency."""
    rng = np.random.default_rng(seed)
    tickers = [c for c in df.columns if c != "VXN"]
    idx = tickers[0]
    constituents = tickers[1:]
    n = len(df)

    for t in tickers:
        if f"{t}_ret" not in df.columns:
            df[f"{t}_ret"] = np.log(df[t]).diff()

    idx_ret = df[f"{idx}_ret"].values
    basket_ret_vals = df[[f"{t}_ret" for t in constituents]].mean(axis=1).values

    # Random entries: ~same frequency as strategy (~18 trades over 1255 days)
    entry_prob = 18.0 / 1255.0
    avg_hold = 7

    position = 0
    hold_days = 0
    equity = 1.0
    daily_rets = []
    trade_pnls = []
    current_pnl = 0.0
    n_trades = 0

    for i in range(1, n):
        dpnl = 0.0

        if position == -1:
            hold_days += 1
            if hold_days >= avg_hold + rng.integers(-2, 5):
                dpnl -= (config.stock_slippage + config.options_slippage)
                trade_pnls.append(current_pnl + dpnl)
                position = 0
                hold_days = 0
                current_pnl = 0.0

        if position == 0 and rng.random() < entry_prob:
            position = -1
            hold_days = 1
            n_trades += 1
            entry_cost = config.stock_slippage + config.options_slippage
            dpnl -= entry_cost
            current_pnl = -entry_cost

        if position == -1 and hold_days > 0:
            ir = idx_ret[i] if not np.isnan(idx_ret[i]) else 0.0
            br = basket_ret_vals[i] if not np.isnan(basket_ret_vals[i]) else 0.0
            und_pnl = -config.underlying_weight * (0.5 * ir + 0.5 * br)
            theta_pnl = -config.options_weight * config.theta_daily_long
            gamma_pnl = config.options_weight * 0.5 * max(0, br**2 - (0.3/np.sqrt(252))**2) * 100
            day_pnl = und_pnl + theta_pnl + gamma_pnl
            dpnl += day_pnl
            current_pnl += day_pnl

        equity *= (1 + dpnl)
        daily_rets.append(dpnl)

    if position == -1:
        trade_pnls.append(current_pnl)

    rets = np.array(daily_rets)
    ann_ret = np.mean(rets) * 252
    ann_vol = np.std(rets) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    return {"sharpe": sharpe, "total_return": equity - 1, "n_trades": n_trades}


# ============================================================
# MAIN
# ============================================================

def main():
    config = MCConfig()

    print("=" * 75)
    print("  MONTE CARLO SIMULATION — COMPLACENCY FADE STRATEGY")
    print(f"  {config.n_simulations} simulations x {config.n_days} days each")
    print("=" * 75)

    # Step 1: Estimate parameters from real data
    print("\n--- Step 1: Estimating Market Parameters ---")
    params = estimate_params_from_real_data()

    # Step 2: Run Monte Carlo
    print(f"\n--- Step 2: Running {config.n_simulations} Simulations ---")
    strategy_results = []
    random_results = []

    for sim in range(config.n_simulations):
        if (sim + 1) % 100 == 0:
            print(f"  Simulation {sim + 1}/{config.n_simulations}...")

        # Generate synthetic path
        df = generate_synthetic_path(params, config.n_days, seed=sim * 7 + 42)

        # Run strategy
        result = run_strategy_on_path(df.copy(), config)
        strategy_results.append(result)

        # Run random baseline
        random_result = run_random_strategy(df.copy(), config, seed=sim * 13 + 99)
        random_results.append(random_result)

    # Step 3: Analyze results
    print("\n--- Step 3: Results ---")

    strat_sharpes = np.array([r["sharpe"] for r in strategy_results])
    strat_returns = np.array([r["total_return"] for r in strategy_results])
    strat_max_dd = np.array([r["max_dd"] for r in strategy_results])
    strat_trades = np.array([r["n_trades"] for r in strategy_results])
    strat_win_rates = np.array([r["win_rate"] for r in strategy_results])
    strat_pf = np.array([r["profit_factor"] for r in strategy_results])

    rand_sharpes = np.array([r["sharpe"] for r in random_results])
    rand_returns = np.array([r["total_return"] for r in random_results])

    print("\n" + "=" * 75)
    print("  MONTE CARLO RESULTS — STRATEGY vs RANDOM BASELINE")
    print("=" * 75)

    print("\n  --- Strategy Performance Distribution ---")
    print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8} {'5th%':>8} {'95th%':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Sharpe Ratio':<25} {strat_sharpes.mean():>8.2f} {np.median(strat_sharpes):>8.2f} "
          f"{strat_sharpes.std():>8.2f} {np.percentile(strat_sharpes, 5):>8.2f} {np.percentile(strat_sharpes, 95):>8.2f}")
    print(f"  {'Total Return':<25} {strat_returns.mean():>7.1%} {np.median(strat_returns):>7.1%} "
          f"{strat_returns.std():>7.1%} {np.percentile(strat_returns, 5):>7.1%} {np.percentile(strat_returns, 95):>7.1%}")
    print(f"  {'Max Drawdown':<25} {strat_max_dd.mean():>7.1%} {np.median(strat_max_dd):>7.1%} "
          f"{strat_max_dd.std():>7.1%} {np.percentile(strat_max_dd, 5):>7.1%} {np.percentile(strat_max_dd, 95):>7.1%}")
    print(f"  {'Trades':<25} {strat_trades.mean():>8.1f} {np.median(strat_trades):>8.1f} "
          f"{strat_trades.std():>8.1f} {np.percentile(strat_trades, 5):>8.1f} {np.percentile(strat_trades, 95):>8.1f}")
    print(f"  {'Win Rate':<25} {strat_win_rates.mean():>7.1%} {np.median(strat_win_rates):>7.1%} "
          f"{strat_win_rates.std():>7.1%} {np.percentile(strat_win_rates, 5):>7.1%} {np.percentile(strat_win_rates, 95):>7.1%}")
    print(f"  {'Profit Factor':<25} {np.clip(strat_pf, 0, 10).mean():>8.2f} {np.clip(np.median(strat_pf), 0, 10):>8.2f} "
          f"{np.clip(strat_pf, 0, 10).std():>8.2f} {np.clip(np.percentile(strat_pf, 5), 0, 10):>8.2f} {np.clip(np.percentile(strat_pf, 95), 0, 10):>8.2f}")

    print("\n  --- Random Baseline Distribution ---")
    print(f"  {'Sharpe Ratio':<25} {rand_sharpes.mean():>8.2f} {np.median(rand_sharpes):>8.2f} "
          f"{rand_sharpes.std():>8.2f} {np.percentile(rand_sharpes, 5):>8.2f} {np.percentile(rand_sharpes, 95):>8.2f}")
    print(f"  {'Total Return':<25} {rand_returns.mean():>7.1%} {np.median(rand_returns):>7.1%} "
          f"{rand_returns.std():>7.1%} {np.percentile(rand_returns, 5):>7.1%} {np.percentile(rand_returns, 95):>7.1%}")

    # Statistical tests
    print("\n  --- Statistical Significance ---")

    # t-test: is strategy Sharpe different from zero?
    t_stat_zero, p_val_zero = stats.ttest_1samp(strat_sharpes, 0)
    print(f"  H0: Strategy Sharpe = 0")
    print(f"    t-statistic: {t_stat_zero:.2f}")
    print(f"    p-value: {p_val_zero:.4f}")
    print(f"    Result: {'REJECT H0 (edge is real)' if p_val_zero < 0.05 else 'CANNOT REJECT H0 (edge may be noise)'}")

    # t-test: is strategy Sharpe different from random?
    t_stat_rand, p_val_rand = stats.ttest_ind(strat_sharpes, rand_sharpes)
    print(f"\n  H0: Strategy Sharpe = Random Sharpe")
    print(f"    t-statistic: {t_stat_rand:.2f}")
    print(f"    p-value: {p_val_rand:.4f}")
    print(f"    Result: {'REJECT H0 (strategy beats random)' if p_val_rand < 0.05 else 'CANNOT REJECT H0 (no better than random)'}")

    # Probability of profit
    prob_profit = np.mean(strat_returns > 0)
    prob_positive_sharpe = np.mean(strat_sharpes > 0)
    prob_sharpe_above_05 = np.mean(strat_sharpes > 0.5)
    prob_beat_random = np.mean(strat_sharpes > np.median(rand_sharpes))

    print(f"\n  --- Probability Metrics ---")
    print(f"  P(profitable):         {prob_profit:.1%}")
    print(f"  P(Sharpe > 0):         {prob_positive_sharpe:.1%}")
    print(f"  P(Sharpe > 0.5):       {prob_sharpe_above_05:.1%}")
    print(f"  P(beats random):       {prob_beat_random:.1%}")

    # Tail risk
    print(f"\n  --- Tail Risk (worst outcomes) ---")
    worst_5 = np.percentile(strat_returns, 5)
    worst_1 = np.percentile(strat_returns, 1)
    worst_dd = np.percentile(strat_max_dd, 5)
    print(f"  5th percentile return:  {worst_5:.1%}")
    print(f"  1st percentile return:  {worst_1:.1%}")
    print(f"  5th percentile max DD:  {worst_dd:.1%}")

    # Simulations where strategy had 0 trades
    zero_trade_sims = np.sum(strat_trades == 0)
    print(f"\n  Simulations with 0 trades: {zero_trade_sims} ({zero_trade_sims/config.n_simulations:.1%})")

    print("\n" + "=" * 75)
    print("  VERDICT")
    print("=" * 75)
    if p_val_zero < 0.05 and p_val_rand < 0.05 and prob_profit > 0.55:
        print("  The strategy shows STATISTICALLY SIGNIFICANT edge across random conditions.")
        print(f"  Expected Sharpe: {strat_sharpes.mean():.2f} ({prob_positive_sharpe:.0%} chance of positive)")
        print(f"  But be aware of tail risk: {-worst_dd:.0%} max DD in worst 5% of scenarios.")
    elif p_val_zero < 0.10 or prob_profit > 0.50:
        print("  The strategy shows MARGINAL edge — somewhat better than random but")
        print("  not robustly significant. The backtest result may partly be luck.")
        print(f"  Expected Sharpe: {strat_sharpes.mean():.2f} (wide distribution: {np.percentile(strat_sharpes, 10):.2f} to {np.percentile(strat_sharpes, 90):.2f})")
    else:
        print("  The strategy does NOT show significant edge in random conditions.")
        print("  The historical backtest result is likely overfitted or path-dependent.")

    print("=" * 75)

    # Save results
    results_df = pd.DataFrame(strategy_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "monte_carlo_results.csv"), index=False)
    print(f"\n  Full results saved to: {os.path.join(OUTPUT_DIR, 'monte_carlo_results.csv')}")


if __name__ == "__main__":
    main()
