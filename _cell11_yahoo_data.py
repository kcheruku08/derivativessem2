# --- Optional: Yahoo Finance (real prices + VXN; SMH IV via ATM options calibration) ---
try:
    import yfinance as yf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

import pandas as pd
import numpy as np
import scipy.stats as stats

# ========== TOGGLE ==========
# "yahoo"  — real SMH/QQQ returns; QQQ IV from ^VXN; SMH IV = rolling RV × (ATM IV/RV) from latest chain
# "simulated" — original OU IV process
DATA_MODE = "yahoo"
YAHOO_PERIOD = "5y"
# EMA on z: strong smoothing (e.g. 45d half-life) collapses the signal range → no trades.
# 12–18d is still "slower" than raw daily noise but preserves ±1–2σ excursions.
Z_EMA_HALFLIFE = 15
IV_RV_LOOKBACK = 21          # days for realized vol used in SMH IV proxy
OLS_WINDOW_REAL = 60         # longer rolling OLS on real data (slower relative-vol dynamics)

def smooth_z(z, halflife):
    """EMA smooth of z (reduces daily noise on entries/exits)."""
    s = pd.Series(z)
    return s.interpolate(limit_direction="both").ewm(halflife=halflife, adjust=False).mean().values


def load_yahoo_iv_and_returns(period="5y", iv_rv_lb=21):
    """
    Pull real market data for backtest.

    - **Returns:** log returns from Yahoo adjusted closes (realized paths for Regime C).
    - **QQQ implied vol:** CBOE Nasdaq-100 Volatility Index `^VXN` (scaled /100). This is a
      real market implied-vol index for NDX, highly correlated with QQQ options IV.
    - **SMH implied vol:** Yahoo does not ship historical ATM IV time series for free. We
      calibrate **once** from the **latest** SMH option chain (ATM call IV / current 21d RV)
      and multiply that ratio × rolling annualized RV. Same idea as a constant risk-premium
      scaling; for production-grade historical IV, use ORATS / CBOE LiveVol / your vendor.

    """
    raw = yf.download(
        ["SMH", "QQQ", "^VXN"],
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    c = raw["Close"]
    df = pd.DataFrame({"SMH": c["SMH"], "QQQ": c["QQQ"], "VXN": c["^VXN"]}).dropna()
    lr_s = np.log(df["SMH"]).diff()
    lr_q = np.log(df["QQQ"]).diff()
    rv_s = lr_s.rolling(iv_rv_lb).std() * np.sqrt(252)
    qqq_iv = (df["VXN"] / 100.0).values.astype(float)

    t = yf.Ticker("SMH")
    opts = t.options
    if opts is None or len(opts) == 0:
        iv_rv_ratio = 1.12
    else:
        chain = t.option_chain(opts[0])
        spot = float(df["SMH"].iloc[-1])
        calls = chain.calls
        j = int((calls["strike"] - spot).abs().values.argmin())
        atm_iv = float(calls.iloc[j]["impliedVolatility"])
        rv_now = float(rv_s.iloc[-1])
        iv_rv_ratio = atm_iv / max(rv_now, 1e-6)

    smh_iv_series = (rv_s * iv_rv_ratio).clip(0.08, 2.0).bfill().ffill()
    smh_iv = smh_iv_series.values.astype(float)
    smh_ret = lr_s.fillna(0.0).values.astype(float)
    qqq_ret = lr_q.fillna(0.0).values.astype(float)
    return smh_iv, qqq_iv, smh_ret, qqq_ret


def simulate_daily_iv_series(days=504, seed=42):
    """
    Simulates daily ATM IV time series for SMH and QQQ.
    Includes:
     - Shared macro component (correlated moves)
     - Idiosyncratic AI FOMO cycles in SMH
     - QQQ put-skew stickiness
     - Earnings vol spikes every ~63 days
     - One major shock event
    """
    rng = np.random.default_rng(seed)

    # Shared macro vol driver (both surfaces move together in risk-off)
    macro_shock = rng.normal(0, 0.008, days)
    macro_shock[int(days * 0.55)] += 0.06   # one big macro event

    # SMH idiosyncratic FOMO cycles
    t = np.arange(days)
    fomo_cycle = 0.03 * np.sin(2 * np.pi * t / 90)   # ~quarterly cycle
    smh_idio   = rng.normal(0, 0.006, days)

    # Earnings spikes in SMH (NVDA / AMD quarterly)
    for q in range(0, days, 63):
        smh_idio[q:q+2] += rng.normal(0.015, 0.005)

    # QQQ put-skew stickiness — mean-reverts faster
    qqq_idio = rng.normal(0, 0.004, days)

    # Build IV levels via OU mean-reversion around each base
    smh_iv_series = np.zeros(days)
    qqq_iv_series = np.zeros(days)
    smh_iv_series[0] = 0.434
    qqq_iv_series[0] = 0.242

    for i in range(1, days):
        smh_iv_series[i] = (smh_iv_series[i-1]
                            + 0.04 * (0.434 - smh_iv_series[i-1])  # OU reversion
                            + 0.8  * macro_shock[i]                 # macro loading
                            + fomo_cycle[i] * 0.3                   # FOMO cycle
                            + smh_idio[i])
        qqq_iv_series[i] = (qqq_iv_series[i-1]
                            + 0.06 * (0.242 - qqq_iv_series[i-1])
                            + 0.55 * macro_shock[i]                 # lower macro beta
                            + qqq_idio[i])

    smh_iv_series = np.clip(smh_iv_series, 0.10, 1.20)
    qqq_iv_series = np.clip(qqq_iv_series, 0.08, 0.80)
    return smh_iv_series, qqq_iv_series


def compute_residual_zscore(smh_iv_s, qqq_iv_s, window=30):
    """
    Rolling OLS regression of SMH IV on QQQ IV.
    Returns the standardised residual — pure idiosyncratic signal.
    """
    n = len(smh_iv_s)
    residuals = np.full(n, np.nan)

    for i in range(window, n):
        x = qqq_iv_s[i-window:i]
        y = smh_iv_s[i-window:i]
        slope, intercept, *_ = stats.linregress(x, y)
        fitted   = slope * qqq_iv_s[i] + intercept
        resid    = smh_iv_s[i] - fitted
        # Z-score using rolling window std of residuals
        past_resids = np.array([
            smh_iv_s[j] - (stats.linregress(
                qqq_iv_s[j-window:j], smh_iv_s[j-window:j]
            )[0] * qqq_iv_s[j] + stats.linregress(
                qqq_iv_s[j-window:j], smh_iv_s[j-window:j]
            )[1])
            for j in range(max(window, i-window), i)
        ])
        if len(past_resids) > 3 and past_resids.std() > 0:
            residuals[i] = resid / past_resids.std()

    return residuals


# --- faster version using pre-computed rolling residuals ---
def compute_residual_zscore_fast(smh_iv_s, qqq_iv_s, window=30):
    n   = len(smh_iv_s)
    raw = np.full(n, np.nan)

    for i in range(window, n):
        x = qqq_iv_s[i-window:i]
        y = smh_iv_s[i-window:i]
        slope, intercept, *_ = stats.linregress(x, y)
        raw[i] = smh_iv_s[i] - (slope * qqq_iv_s[i] + intercept)

    # Z-score the raw residuals with a rolling 30d std
    z = np.full(n, np.nan)
    for i in range(window * 2, n):
        window_resids = raw[i-window:i]
        mu, sig = np.nanmean(window_resids), np.nanstd(window_resids)
        if sig > 0:
            z[i] = (raw[i] - mu) / sig
    return z, raw


if DATA_MODE == "yahoo":
    smh_iv_ts, qqq_iv_ts, smh_ret, qqq_ret = load_yahoo_iv_and_returns(
        period=YAHOO_PERIOD, iv_rv_lb=IV_RV_LOOKBACK
    )
    DAYS = len(smh_iv_ts)
    OLS_WINDOW = OLS_WINDOW_REAL
    MIN_HOLD_DAYS_AB = 21
    z_score, raw_resid = compute_residual_zscore_fast(smh_iv_ts, qqq_iv_ts, window=OLS_WINDOW)
    z_score = smooth_z(z_score, Z_EMA_HALFLIFE)
    # Thresholds tuned for EMA-smoothed z (smaller range than raw z)
    ENTRY_THRESH = 1.0
    EXIT_THRESH = 0.35
    CONV_STD_THRESH = 1.35
    spread_smooth = 60
else:
    DAYS = 504   # ~2 trading years
    OLS_WINDOW = 30
    MIN_HOLD_DAYS_AB = 0
    smh_iv_ts, qqq_iv_ts = simulate_daily_iv_series(DAYS)
    z_score, raw_resid   = compute_residual_zscore_fast(smh_iv_ts, qqq_iv_ts, window=OLS_WINDOW)
    ENTRY_THRESH = 2.0
    EXIT_THRESH = 0.5
    CONV_STD_THRESH = 1.5
    spread_smooth = 30
    smh_ret, qqq_ret = simulate_price_returns(smh_iv_ts, qqq_iv_ts)

print(f"DATA_MODE = {DATA_MODE}  |  DAYS = {DAYS}  |  OLS_WINDOW = {OLS_WINDOW}")
print(f"SMH IV range:   {np.nanmin(smh_iv_ts):.1%} – {np.nanmax(smh_iv_ts):.1%}")
print(f"QQQ IV range:   {np.nanmin(qqq_iv_ts):.1%} – {np.nanmax(qqq_iv_ts):.1%}")
print(f"Z-score range:  {np.nanmin(z_score):.2f} – {np.nanmax(z_score):.2f}")
print(f"ENTRY/EXIT (σ): {ENTRY_THRESH} / {EXIT_THRESH}  |  min hold A/B: {MIN_HOLD_DAYS_AB}d")

if DATA_MODE == "yahoo":
    print(f"SMH ann. realized vol: {np.std(smh_ret)*np.sqrt(252):.1%}")
    print(f"QQQ ann. realized vol: {np.std(qqq_ret)*np.sqrt(252):.1%}")
    print(f"Realized corr(SMH, QQQ): {np.corrcoef(smh_ret, qqq_ret)[0,1]:.2f}")
else:
    print(f'SMH annualised realized vol: {smh_ret.std()*np.sqrt(252):.1%}  '
          f'(implied avg: {smh_iv_ts.mean():.1%})')
    print(f'QQQ annualised realized vol: {qqq_ret.std()*np.sqrt(252):.1%}  '
          f'(implied avg: {qqq_iv_ts.mean():.1%})')
    print(f'Realized corr(SMH, QQQ):     {np.corrcoef(smh_ret, qqq_ret)[0,1]:.2f}')
print("✓ IV series and returns ready")

def backtest_strategy(z_scores, smh_rets, qqq_rets, smh_ivs, qqq_ivs, entry_thresh, exit_thresh, min_hold, opt_slippage=0.015, opt_daily_carry=0.0003, stock_slippage=0.001):
    """
    Backtests an Underlying Pairs trade and a Delta-Neutral Options Pairs trade,
    incorporating realistic market frictions (slippage and delta-hedging/carry costs).
    """
    n = len(z_scores)
    positions = np.zeros(n)
    pos = 0
    hold_days = 0
    
    for i in range(1, n):
        z = z_scores[i-1] # Trade on yesterday's signal to avoid lookahead bias
        
        if np.isnan(z):
            positions[i] = 0
            continue
            
        if pos == -1 and z < exit_thresh and hold_days >= min_hold:
            pos = 0
            hold_days = 0
        elif pos == 1 and z > -exit_thresh and hold_days >= min_hold:
            pos = 0
            hold_days = 0
            
        if pos == 0:
            if z > entry_thresh:
                pos = -1
                hold_days = 1
            elif z < -entry_thresh:
                pos = 1
                hold_days = 1
        else:
            hold_days += 1
            
        positions[i] = pos
        
    trade_activity = np.abs(np.diff(np.append([0], positions)))
    is_in_market = np.abs(positions) > 0
        
    # 1. Underlying ETF Returns
    underlying_rets = 0.5 * positions * (smh_rets - qqq_rets)
    # Apply stock slippage (e.g. 10bps when entering/exiting)
    underlying_rets -= (trade_activity * stock_slippage)
    
    # 2. Options Strategy Returns (Proxy using daily % change in IV)
    smh_iv_pct_change = np.zeros(n)
    qqq_iv_pct_change = np.zeros(n)
    smh_iv_pct_change[1:] = (smh_ivs[1:] - smh_ivs[:-1]) / smh_ivs[:-1]
    qqq_iv_pct_change[1:] = (qqq_ivs[1:] - qqq_ivs[:-1]) / qqq_ivs[:-1]
    
    options_rets = 0.5 * positions * (smh_iv_pct_change - qqq_iv_pct_change)
    
    # Apply Realistic Options Frictions
    # A. Transaction Costs (Bid/Ask Slippage): Straddles have wide spreads. We penalize entries and exits.
    options_rets -= (trade_activity * opt_slippage) 
    
    # B. Daily Carry / Hedging Cost: Holding delta-neutral options incurs continuous drag (un-matched theta, borrow, gamma scalping losses)
    options_rets -= (is_in_market * opt_daily_carry)
    
    def calc_metrics(ret_array):
        cum_returns = np.cumprod(1 + ret_array)
        ann_ret = np.mean(ret_array) * 252
        ann_vol = np.std(ret_array) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        running_max = np.maximum.accumulate(cum_returns)
        dd = (cum_returns - running_max) / running_max
        max_dd = np.min(dd) if len(dd) > 0 else 0
        return {
            "Annualized Return": f"{ann_ret:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Total Return": f"{cum_returns[-1] - 1.0:.2%}",
            "Days in Market": f"{np.count_nonzero(positions)} / {n}"
        }
        
    return calc_metrics(underlying_rets), calc_metrics(options_rets)

# Run the backtest using previously computed variables
und_metrics, opt_metrics = backtest_strategy(
    z_score, smh_ret, qqq_ret, smh_iv_ts, qqq_iv_ts, ENTRY_THRESH, EXIT_THRESH, MIN_HOLD_DAYS_AB
)

print("-" * 40)
print("Backtest Results (Underlying ETFs - Shares)")
for k, v in und_metrics.items():
    print(f"{k}: {v}")
print("-" * 40)
print("Backtest Results (Options Straddles - IV proxy)")
for k, v in opt_metrics.items():
    print(f"{k}: {v}")
print("-" * 40)
