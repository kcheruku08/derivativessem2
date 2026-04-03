# --- Optional: Yahoo Finance (real prices + VXN; SMH IV from lagged premium proxy) ---
try:
    import yfinance as yf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats

# Ensure Unicode prints work on Windows terminals (cp1252 default can fail).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Keep yfinance cache in a writable project-local folder.
YF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".yfinance_cache")
os.makedirs(YF_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YF_CACHE_DIR)

# ========== TOGGLE ==========
# "yahoo"  - real SMH/QQQ returns; QQQ IV from ^VXN; SMH IV = rolling RV x lagged QQQ premium proxy
# "simulated" — original OU IV process
DATA_MODE = "simulated"
BACKTEST_YEARS = 5
YAHOO_PERIOD = f"{BACKTEST_YEARS}y"
# EMA on z: strong smoothing (e.g. 45d half-life) collapses the signal range → no trades.
# 12–18d is still "slower" than raw daily noise but preserves ±1–2σ excursions.
Z_EMA_HALFLIFE = 15
IV_RV_LOOKBACK = 21          # days for realized vol used in SMH IV proxy
OLS_WINDOW_REAL = 60         # longer rolling OLS on real data (slower relative-vol dynamics)

# Unified signal settings across data modes (prevents mode-dependent signal behavior).
SIGNAL_OLS_WINDOW = 60
SIGNAL_Z_EMA_HALFLIFE = 15
SIGNAL_ENTRY_THRESH = 1.0
SIGNAL_EXIT_THRESH = 0.35
SIGNAL_MIN_HOLD_DAYS = 21
USE_ASYMMETRIC_ZSCORE = True

# Simulated-only override to increase time in market.
SIM_ENTRY_THRESH = 0.75
SIM_EXIT_THRESH = 0.25
SIM_MIN_HOLD_DAYS = 21

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
      estimate SMH IV from SMH rolling realized vol times a QQQ-derived risk-premium factor.
      The premium factor is lagged by one day so each timestamp only uses information
      available up to t-1 (no look-ahead bias).

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
    rv_q = lr_q.rolling(iv_rv_lb).std() * np.sqrt(252)
    qqq_iv = (df["VXN"] / 100.0).values.astype(float)

    # QQQ implied/realized premium proxy, lagged by one day to avoid look-ahead.
    qqq_premium = (df["VXN"] / 100.0) / rv_q.replace(0.0, np.nan)
    lagged_premium = (
        qqq_premium.replace([np.inf, -np.inf], np.nan)
        .ewm(halflife=63, adjust=False)
        .mean()
        .shift(1)
        .fillna(1.12)
        .clip(0.70, 2.00)
    )

    # Warm-up fallback uses same-day QQQ IV scaling (observable on date t).
    warmup_proxy = ((df["VXN"] / 100.0) * 1.15).clip(0.08, 2.0)
    smh_iv_series = (rv_s * lagged_premium).clip(0.08, 2.0).fillna(warmup_proxy).ffill()
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
                            + 0.05 * (0.434 - smh_iv_series[i-1])  # OU reversion
                            + 0.45 * macro_shock[i]                 # macro loading
                            + fomo_cycle[i] * 0.12                  # FOMO cycle
                            + 0.6 * smh_idio[i])
        qqq_iv_series[i] = (qqq_iv_series[i-1]
                            + 0.08 * (0.242 - qqq_iv_series[i-1])
                            + 0.35 * macro_shock[i]                 # lower macro beta
                            + 0.7 * qqq_idio[i])

    smh_iv_series = np.clip(smh_iv_series, 0.10, 1.20)
    qqq_iv_series = np.clip(qqq_iv_series, 0.08, 0.80)
    return smh_iv_series, qqq_iv_series


def simulate_price_returns(smh_iv_s, qqq_iv_s, corr=0.72, seed=123):
    """
    Simulate daily log returns with time-varying vol informed by implied vol levels.
    """
    n = len(smh_iv_s)
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0.0, 1.0, n)
    z2 = rng.normal(0.0, 1.0, n)
    zq = corr * z1 + np.sqrt(max(1.0 - corr**2, 0.0)) * z2

    # Use IV as volatility proxy, with a moderate risk-premium discount for realized vol.
    smh_sigma = np.asarray(smh_iv_s, dtype=float) / np.sqrt(252.0) * 0.90
    qqq_sigma = np.asarray(qqq_iv_s, dtype=float) / np.sqrt(252.0) * 0.90

    smh_ret = smh_sigma * z1
    qqq_ret = qqq_sigma * zq
    return smh_ret.astype(float), qqq_ret.astype(float)


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


def compute_residual_zscore_asymmetric_fast(smh_iv_s, qqq_iv_s, window=30, min_side_obs=5):
    """
    Asymmetric residual z-score:
    - positive residuals are scaled by rolling std of positive residual history
    - negative residuals are scaled by rolling std of negative residual history
    Falls back to symmetric std when one side has too few observations.
    """
    z_sym, raw = compute_residual_zscore_fast(smh_iv_s, qqq_iv_s, window=window)
    n = len(raw)
    z_asym = np.full(n, np.nan)

    for i in range(window * 2, n):
        resid_now = raw[i]
        if np.isnan(resid_now):
            continue

        window_resids = raw[i-window:i]
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
            z_asym[i] = (resid_now - mu) / sig_side

    # Preserve early warmup behavior from symmetric z.
    z_asym[: window * 2] = z_sym[: window * 2]
    return z_asym, raw


if DATA_MODE == "yahoo":
    smh_iv_ts, qqq_iv_ts, smh_ret, qqq_ret = load_yahoo_iv_and_returns(
        period=YAHOO_PERIOD, iv_rv_lb=IV_RV_LOOKBACK
    )
    DAYS = len(smh_iv_ts)
    OLS_WINDOW = SIGNAL_OLS_WINDOW
    MIN_HOLD_DAYS_AB = SIGNAL_MIN_HOLD_DAYS
    if USE_ASYMMETRIC_ZSCORE:
        z_score, raw_resid = compute_residual_zscore_asymmetric_fast(smh_iv_ts, qqq_iv_ts, window=OLS_WINDOW)
    else:
        z_score, raw_resid = compute_residual_zscore_fast(smh_iv_ts, qqq_iv_ts, window=OLS_WINDOW)
    z_score = smooth_z(z_score, SIGNAL_Z_EMA_HALFLIFE)
    ENTRY_THRESH = SIGNAL_ENTRY_THRESH
    EXIT_THRESH = SIGNAL_EXIT_THRESH
    CONV_STD_THRESH = 1.35
    spread_smooth = 60
else:
    DAYS = 252 * BACKTEST_YEARS
    OLS_WINDOW = SIGNAL_OLS_WINDOW
    MIN_HOLD_DAYS_AB = SIM_MIN_HOLD_DAYS
    smh_iv_ts, qqq_iv_ts = simulate_daily_iv_series(DAYS)
    if USE_ASYMMETRIC_ZSCORE:
        z_score, raw_resid = compute_residual_zscore_asymmetric_fast(smh_iv_ts, qqq_iv_ts, window=OLS_WINDOW)
    else:
        z_score, raw_resid = compute_residual_zscore_fast(smh_iv_ts, qqq_iv_ts, window=OLS_WINDOW)
    z_score = smooth_z(z_score, SIGNAL_Z_EMA_HALFLIFE)
    ENTRY_THRESH = SIM_ENTRY_THRESH
    EXIT_THRESH = SIM_EXIT_THRESH
    CONV_STD_THRESH = 1.35
    spread_smooth = 60
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

def backtest_strategy(
    z_scores, smh_rets, qqq_rets, smh_ivs, qqq_ivs,
    entry_thresh, exit_thresh, min_hold,
    opt_slippage=0.01,
    stock_slippage=0.001,
    smh_vega=1.25, qqq_vega=1.00,
    smh_theta=0.00022, qqq_theta=0.00018,
    smh_gamma=0.10, qqq_gamma=0.08,
    hedge_ratio=1.0,
    reentry_buffer=0.15,
    cooldown_days=3
):
    n = len(z_scores)
    positions = np.zeros(n)
    pos = 0
    hold_days = 0
    cooldown = 0

    for i in range(1, n):
        z = z_scores[i-1]

        if np.isnan(z):
            positions[i] = 0
            continue

        if pos == -1 and z < exit_thresh and hold_days >= min_hold:
            pos = 0
            hold_days = 0
            cooldown = cooldown_days
        elif pos == 1 and z > -exit_thresh and hold_days >= min_hold:
            pos = 0
            hold_days = 0
            cooldown = cooldown_days

        if pos == 0:
            if cooldown > 0:
                cooldown -= 1
            else:
                enter_short_level = entry_thresh + reentry_buffer
                enter_long_level = -entry_thresh - reentry_buffer
                if z > enter_short_level:
                    pos = -1
                    hold_days = 1
                elif z < enter_long_level:
                    pos = 1
                    hold_days = 1
        else:
            hold_days += 1

        positions[i] = pos

    trade_activity = np.abs(np.diff(np.append([0], positions)))

    # Underlying version unchanged
    underlying_rets = 0.5 * positions * (smh_rets - qqq_rets)
    underlying_rets -= trade_activity * stock_slippage

    # --- improved options spread proxy ---

    # absolute IV point changes, not percent changes
    d_smh_iv = np.zeros(n)
    d_qqq_iv = np.zeros(n)
    d_smh_iv[1:] = smh_ivs[1:] - smh_ivs[:-1]
    d_qqq_iv[1:] = qqq_ivs[1:] - qqq_ivs[:-1]

    # crude realized-move proxy for gamma benefit/cost
    smh_move = np.abs(smh_rets)
    qqq_move = np.abs(qqq_rets)

    # side conventions from relative-vol signal
    # pos = +1 => long SMH vol, short QQQ vol
    # pos = -1 => short SMH vol, long QQQ vol
    smh_side = positions
    qqq_side = -positions

    # theta is negative for long option legs, positive for short option legs
    smh_leg = (
        smh_side * smh_vega * d_smh_iv
        + smh_side * smh_gamma * smh_move
        - np.sign(smh_side) * smh_theta * (np.abs(smh_side) > 0)
    )

    qqq_leg = (
        qqq_side * hedge_ratio * qqq_vega * d_qqq_iv
        + qqq_side * hedge_ratio * qqq_gamma * qqq_move
        - np.sign(qqq_side) * qqq_theta * (np.abs(qqq_side) > 0)
    )

    options_rets = smh_leg + qqq_leg

    # slippage only when position changes
    options_rets -= trade_activity * opt_slippage

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


