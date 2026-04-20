#!/usr/bin/env python3
"""
Volatility Surface Arbitrage Research: AI Semiconductor Stocks vs NASDAQ (QQQ)
==============================================================================
Investigates dispersion trading opportunity between a basket of AI semi stocks
(NVDA, AVGO, TSM) and QQQ/NASDAQ index options.

Pulls live options chains via yfinance, builds IV term structures, computes
basket vs index spread, and analyzes divergences.

Author: Research Script
Date: April 2026
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import interpolate
import sys
import os

OUTPUT_DIR = "/Users/krishicherukupalli/derivativessem2/backtest_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
SEMI_TICKERS = ["NVDA", "AVGO", "TSM"]
INDEX_TICKER = "QQQ"
ALL_TICKERS = SEMI_TICKERS + [INDEX_TICKER]

# Equal-weight basket (can switch to market-cap weight below)
BASKET_WEIGHTS = {"NVDA": 1/3, "AVGO": 1/3, "TSM": 1/3}

# ── Helper Functions ───────────────────────────────────────────────────────────

def get_spot_price(ticker_obj):
    """Get current spot price from yfinance ticker."""
    try:
        info = ticker_obj.info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if price:
            return float(price)
    except Exception:
        pass
    # fallback: last close from history
    try:
        hist = ticker_obj.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def find_atm_iv(chain_calls, chain_puts, spot):
    """
    Given calls and puts DataFrames and a spot price, find the ATM implied vol.
    Uses the strike closest to spot, averaging call and put IV.
    """
    if chain_calls.empty and chain_puts.empty:
        return np.nan

    # Use calls for ATM strike selection
    df = chain_calls if not chain_calls.empty else chain_puts
    df = df.dropna(subset=["impliedVolatility"])
    if df.empty:
        return np.nan

    # Find strike closest to spot
    atm_strike = df.loc[(df["strike"] - spot).abs().idxmin(), "strike"]

    ivs = []
    for chain in [chain_calls, chain_puts]:
        if chain.empty:
            continue
        row = chain[chain["strike"] == atm_strike]
        if not row.empty:
            iv = row["impliedVolatility"].iloc[0]
            if iv > 0.001:  # filter garbage
                ivs.append(iv)

    return np.mean(ivs) if ivs else np.nan


def compute_skew_metrics(chain_calls, chain_puts, spot):
    """
    Compute 25-delta skew proxy: IV at 95% moneyness minus IV at 105% moneyness (puts vs calls).
    Also returns the full smile data for plotting.
    """
    results = {"skew_25d": np.nan, "smile_strikes": [], "smile_ivs": []}

    # Combine and use puts for downside, calls for upside
    put_df = chain_puts.dropna(subset=["impliedVolatility"]).copy() if not chain_puts.empty else pd.DataFrame()
    call_df = chain_calls.dropna(subset=["impliedVolatility"]).copy() if not chain_calls.empty else pd.DataFrame()

    if put_df.empty and call_df.empty:
        return results

    # Build combined smile: use puts below ATM, calls above ATM
    combined = []
    if not put_df.empty:
        below = put_df[put_df["strike"] <= spot]
        for _, row in below.iterrows():
            combined.append((row["strike"], row["impliedVolatility"]))
    if not call_df.empty:
        above = call_df[call_df["strike"] >= spot]
        for _, row in above.iterrows():
            combined.append((row["strike"], row["impliedVolatility"]))

    if len(combined) < 3:
        return results

    combined = sorted(combined, key=lambda x: x[0])
    strikes = [c[0] for c in combined]
    ivs = [c[1] for c in combined]

    results["smile_strikes"] = strikes
    results["smile_ivs"] = ivs

    # 25-delta skew proxy: IV at 90% spot vs IV at 110% spot
    target_low = spot * 0.90
    target_high = spot * 1.10

    try:
        f = interpolate.interp1d(strikes, ivs, kind="linear", fill_value="extrapolate")
        iv_low = float(f(target_low))
        iv_high = float(f(target_high))
        results["skew_25d"] = iv_low - iv_high
    except Exception:
        pass

    return results


# ── Main Data Pull ─────────────────────────────────────────────────────────────

def pull_all_data():
    """Pull options chains for all tickers, compute ATM IV term structures."""

    print("=" * 80)
    print("VOLATILITY SURFACE RESEARCH: AI Semiconductors vs NASDAQ")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Tickers: {', '.join(ALL_TICKERS)}")
    print()

    # Store results
    term_structures = {}   # ticker -> DataFrame with columns [expiry, dte, atm_iv]
    skew_data = {}         # ticker -> DataFrame with columns [expiry, dte, skew_25d]
    spot_prices = {}
    smile_snapshots = {}   # ticker -> dict of expiry -> (strikes, ivs)

    for ticker_sym in ALL_TICKERS:
        print(f"--- Pulling data for {ticker_sym} ---")
        tk = yf.Ticker(ticker_sym)

        spot = get_spot_price(tk)
        if spot is None:
            print(f"  WARNING: Could not get spot price for {ticker_sym}, skipping.")
            continue
        spot_prices[ticker_sym] = spot
        print(f"  Spot: ${spot:.2f}")

        expirations = tk.options
        if not expirations:
            print(f"  WARNING: No options expirations for {ticker_sym}")
            continue
        print(f"  Expirations available: {len(expirations)}")

        rows_ts = []
        rows_skew = []
        smile_dict = {}

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_date - datetime.now()).days
                if dte < 1:
                    continue  # skip expired

                chain = tk.option_chain(exp_str)
                calls = chain.calls
                puts = chain.puts

                atm_iv = find_atm_iv(calls, puts, spot)
                skew_info = compute_skew_metrics(calls, puts, spot)

                rows_ts.append({
                    "expiry": exp_date,
                    "dte": dte,
                    "atm_iv": atm_iv
                })
                rows_skew.append({
                    "expiry": exp_date,
                    "dte": dte,
                    "skew_25d": skew_info["skew_25d"]
                })
                if skew_info["smile_strikes"]:
                    smile_dict[exp_str] = (skew_info["smile_strikes"], skew_info["smile_ivs"])

            except Exception as e:
                # Silently skip problematic expirations
                continue

        if rows_ts:
            df_ts = pd.DataFrame(rows_ts).sort_values("dte").reset_index(drop=True)
            df_ts = df_ts.dropna(subset=["atm_iv"])
            term_structures[ticker_sym] = df_ts
            print(f"  Term structure points: {len(df_ts)}")

            df_skew = pd.DataFrame(rows_skew).sort_values("dte").reset_index(drop=True)
            skew_data[ticker_sym] = df_skew

            smile_snapshots[ticker_sym] = smile_dict
        else:
            print(f"  WARNING: No valid term structure data for {ticker_sym}")

        print()

    return term_structures, skew_data, spot_prices, smile_snapshots


def compute_basket_iv(term_structures, weights=BASKET_WEIGHTS):
    """
    Compute a weighted-average basket IV from the semi stocks.
    Interpolates to common DTE grid so we can average across names.
    """
    # Find common DTE range
    all_dtes = set()
    for sym in SEMI_TICKERS:
        if sym in term_structures:
            all_dtes.update(term_structures[sym]["dte"].tolist())

    if not all_dtes:
        return pd.DataFrame()

    common_dtes = sorted(all_dtes)

    basket_rows = []
    for dte in common_dtes:
        weighted_iv = 0.0
        total_weight = 0.0
        for sym in SEMI_TICKERS:
            if sym not in term_structures:
                continue
            df = term_structures[sym]
            # Find closest DTE
            idx = (df["dte"] - dte).abs().idxmin()
            row = df.loc[idx]
            if abs(row["dte"] - dte) <= 7 and not np.isnan(row["atm_iv"]):
                weighted_iv += weights[sym] * row["atm_iv"]
                total_weight += weights[sym]

        if total_weight > 0:
            basket_rows.append({
                "dte": dte,
                "basket_iv": weighted_iv / total_weight
            })

    return pd.DataFrame(basket_rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_term_structures(term_structures, basket_df):
    """Plot ATM IV term structures for all tickers + basket."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    colors = {"NVDA": "#76B900", "AVGO": "#CC0000", "TSM": "#0066CC", "QQQ": "#333333"}
    ax = axes[0]

    for sym in ALL_TICKERS:
        if sym in term_structures:
            df = term_structures[sym]
            ax.plot(df["dte"], df["atm_iv"] * 100, "o-", label=sym,
                    color=colors.get(sym, "gray"), markersize=4, linewidth=1.5)

    if not basket_df.empty:
        ax.plot(basket_df["dte"], basket_df["basket_iv"] * 100, "s--",
                label="Semi Basket (EW)", color="#FF6600", markersize=5, linewidth=2)

    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("ATM Implied Volatility (%)")
    ax.set_title("ATM IV Term Structure: AI Semiconductors vs QQQ", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Bottom panel: spread
    ax2 = axes[1]
    if not basket_df.empty and INDEX_TICKER in term_structures:
        qqq_df = term_structures[INDEX_TICKER]
        spread_rows = []
        for _, brow in basket_df.iterrows():
            dte = brow["dte"]
            idx = (qqq_df["dte"] - dte).abs().idxmin()
            qrow = qqq_df.loc[idx]
            if abs(qrow["dte"] - dte) <= 7 and not np.isnan(qrow["atm_iv"]):
                spread_rows.append({
                    "dte": dte,
                    "spread": (brow["basket_iv"] - qrow["atm_iv"]) * 100
                })

        if spread_rows:
            sdf = pd.DataFrame(spread_rows)
            ax2.bar(sdf["dte"], sdf["spread"], width=max(1, sdf["dte"].diff().median() * 0.6),
                    color=np.where(sdf["spread"] > 0, "#FF6600", "#0066CC"), alpha=0.7)
            ax2.axhline(y=0, color="black", linewidth=0.8)
            ax2.set_xlabel("Days to Expiration")
            ax2.set_ylabel("Spread (pp)")
            ax2.set_title("Semi Basket IV minus QQQ IV (percentage points)", fontsize=11)
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "term_structure_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_skew_comparison(skew_data):
    """Plot 25-delta skew across tenors for each ticker."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"NVDA": "#76B900", "AVGO": "#CC0000", "TSM": "#0066CC", "QQQ": "#333333"}

    for sym in ALL_TICKERS:
        if sym in skew_data:
            df = skew_data[sym].dropna(subset=["skew_25d"])
            if not df.empty:
                ax.plot(df["dte"], df["skew_25d"] * 100, "o-", label=sym,
                        color=colors.get(sym, "gray"), markersize=4, linewidth=1.5)

    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Skew: IV(90% moneyness) - IV(110% moneyness) (pp)")
    ax.set_title("Volatility Skew Term Structure (25-delta proxy)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "skew_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_smile_snapshot(smile_snapshots, spot_prices, target_dte=30):
    """Plot vol smile for a near-30-DTE expiry for each ticker."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"NVDA": "#76B900", "AVGO": "#CC0000", "TSM": "#0066CC", "QQQ": "#333333"}

    for sym in ALL_TICKERS:
        if sym not in smile_snapshots or sym not in spot_prices:
            continue
        spot = spot_prices[sym]
        best_exp = None
        best_dte_diff = 999
        for exp_str, (strikes, ivs) in smile_snapshots[sym].items():
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - datetime.now()).days
            if abs(dte - target_dte) < best_dte_diff:
                best_dte_diff = abs(dte - target_dte)
                best_exp = exp_str

        if best_exp is None:
            continue

        strikes, ivs = smile_snapshots[sym][best_exp]
        moneyness = [k / spot for k in strikes]
        ax.plot(moneyness, [iv * 100 for iv in ivs], "-", label=f"{sym} ({best_exp})",
                color=colors.get(sym, "gray"), linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Moneyness (Strike / Spot)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"Volatility Smile (~{target_dte} DTE)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.80, 1.20)
    ax.axvline(x=1.0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "smile_snapshot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Analysis ───────────────────────────────────────────────────────────────────

def print_analysis(term_structures, skew_data, basket_df, spot_prices):
    """Print detailed analysis of vol surface divergences."""

    print("\n" + "=" * 80)
    print("ANALYSIS: VOLATILITY SURFACE COMPARISON")
    print("=" * 80)

    # 1. Spot prices and ATM IV summary
    print("\n1. SPOT PRICES AND SHORT-DATED ATM IV")
    print("-" * 50)
    for sym in ALL_TICKERS:
        if sym in spot_prices and sym in term_structures:
            df = term_structures[sym]
            short = df[df["dte"] <= 45]
            if not short.empty:
                near_iv = short.iloc[0]["atm_iv"]
                print(f"  {sym:6s}  Spot: ${spot_prices[sym]:>10.2f}   Near-term ATM IV: {near_iv*100:5.1f}%")

    # 2. Term structure shape
    print("\n2. TERM STRUCTURE SHAPE")
    print("-" * 50)
    for sym in ALL_TICKERS:
        if sym not in term_structures:
            continue
        df = term_structures[sym]
        short = df[df["dte"] <= 45]["atm_iv"]
        medium = df[(df["dte"] > 45) & (df["dte"] <= 120)]["atm_iv"]
        long_term = df[df["dte"] > 120]["atm_iv"]

        parts = []
        if not short.empty:
            parts.append(f"Short(<45d): {short.mean()*100:.1f}%")
        if not medium.empty:
            parts.append(f"Med(45-120d): {medium.mean()*100:.1f}%")
        if not long_term.empty:
            parts.append(f"Long(>120d): {long_term.mean()*100:.1f}%")

        shape = "FLAT"
        if not short.empty and not long_term.empty:
            if long_term.mean() > short.mean() * 1.03:
                shape = "CONTANGO (upward sloping)"
            elif long_term.mean() < short.mean() * 0.97:
                shape = "BACKWARDATION (downward sloping)"

        print(f"  {sym:6s}  {' | '.join(parts)}")
        print(f"         Shape: {shape}")

    # 3. Basket vs QQQ spread
    print("\n3. BASKET vs QQQ SPREAD ANALYSIS")
    print("-" * 50)
    if not basket_df.empty and INDEX_TICKER in term_structures:
        qqq_df = term_structures[INDEX_TICKER]
        print(f"  {'DTE':>6s}  {'Basket IV':>10s}  {'QQQ IV':>10s}  {'Spread':>10s}  {'Ratio':>8s}")
        print(f"  {'---':>6s}  {'---':>10s}  {'---':>10s}  {'---':>10s}  {'---':>8s}")

        spread_summary = []
        for _, brow in basket_df.iterrows():
            dte = brow["dte"]
            idx = (qqq_df["dte"] - dte).abs().idxmin()
            qrow = qqq_df.loc[idx]
            if abs(qrow["dte"] - dte) > 7 or np.isnan(qrow["atm_iv"]):
                continue

            spread = (brow["basket_iv"] - qrow["atm_iv"]) * 100
            ratio = brow["basket_iv"] / qrow["atm_iv"] if qrow["atm_iv"] > 0 else np.nan
            spread_summary.append({"dte": dte, "spread": spread, "ratio": ratio,
                                   "basket_iv": brow["basket_iv"], "qqq_iv": qrow["atm_iv"]})

            # Print a subset to avoid flooding
            if dte <= 180 or dte % 90 < 15:
                print(f"  {int(dte):6d}  {brow['basket_iv']*100:9.1f}%  {qrow['atm_iv']*100:9.1f}%  "
                      f"{spread:+9.1f}pp  {ratio:7.2f}x")

        if spread_summary:
            sdf = pd.DataFrame(spread_summary)
            print(f"\n  Average spread: {sdf['spread'].mean():+.1f} pp")
            print(f"  Max spread:     {sdf['spread'].max():+.1f} pp (at {sdf.loc[sdf['spread'].idxmax(), 'dte']} DTE)")
            print(f"  Min spread:     {sdf['spread'].min():+.1f} pp (at {sdf.loc[sdf['spread'].idxmin(), 'dte']} DTE)")
            print(f"  Avg ratio:      {sdf['ratio'].mean():.2f}x")
    else:
        print("  Insufficient data for spread analysis.")

    # 4. Skew comparison
    print("\n4. SKEW COMPARISON (90%-110% moneyness)")
    print("-" * 50)
    for sym in ALL_TICKERS:
        if sym in skew_data:
            df = skew_data[sym].dropna(subset=["skew_25d"])
            short_skew = df[df["dte"] <= 60]["skew_25d"]
            if not short_skew.empty:
                print(f"  {sym:6s}  Avg short-dated skew: {short_skew.mean()*100:+.1f} pp")

    # 5. Dispersion trade signal
    print("\n5. DISPERSION TRADE SIGNAL ASSESSMENT")
    print("-" * 50)
    if not basket_df.empty and INDEX_TICKER in term_structures:
        qqq_df = term_structures[INDEX_TICKER]
        # Check short-dated spread
        short_basket = basket_df[basket_df["dte"] <= 60]
        short_qqq = qqq_df[qqq_df["dte"] <= 60]

        if not short_basket.empty and not short_qqq.empty:
            avg_basket = short_basket["basket_iv"].mean()
            avg_qqq = short_qqq["atm_iv"].mean()
            spread_pct = (avg_basket - avg_qqq) / avg_qqq * 100

            print(f"  Short-dated (<60 DTE):")
            print(f"    Basket avg IV: {avg_basket*100:.1f}%")
            print(f"    QQQ avg IV:    {avg_qqq*100:.1f}%")
            print(f"    Spread:        {(avg_basket - avg_qqq)*100:+.1f} pp ({spread_pct:+.1f}% relative)")
            print()

            if avg_basket > avg_qqq * 1.3:
                print("  SIGNAL: Large positive spread -- single-stock vol is RICH vs index.")
                print("  Classic dispersion setup: SELL single-stock straddles, BUY index straddles.")
                print("  (This profits if realized correlation rises / individual moves are less than priced.)")
            elif avg_basket > avg_qqq * 1.1:
                print("  SIGNAL: Moderate positive spread -- typical regime for dispersion.")
                print("  Single-stock vol trades at a normal premium to index vol.")
                print("  Look for names where the premium is extreme vs historical.")
            else:
                print("  SIGNAL: Tight spread -- correlation is priced high.")
                print("  Potential reverse dispersion: BUY single-stock vol, SELL index vol.")
                print("  (This profits if idiosyncratic moves are larger than priced.)")

        # Check term structure divergence
        long_basket = basket_df[basket_df["dte"] > 120]
        long_qqq = qqq_df[qqq_df["dte"] > 120]
        if not long_basket.empty and not long_qqq.empty and not short_basket.empty and not short_qqq.empty:
            short_spread = short_basket["basket_iv"].mean() - short_qqq["atm_iv"].mean()
            long_spread = long_basket["basket_iv"].mean() - long_qqq["atm_iv"].mean()
            print(f"\n  Term structure of spread:")
            print(f"    Short-dated spread: {short_spread*100:+.1f} pp")
            print(f"    Long-dated spread:  {long_spread*100:+.1f} pp")
            if short_spread > long_spread:
                print("    -> Spread is inverted (wider at front). Near-term event risk priced into semis.")
                print("    -> Calendar spread opportunity: sell short-dated semi vol, buy long-dated.")
            else:
                print("    -> Spread widens with tenor. Long-dated uncertainty premium in semis.")

    # 6. Background on dispersion trading
    print("\n6. BACKGROUND: DISPERSION TRADING IN SEMICONDUCTORS")
    print("-" * 50)
    print("""
  Dispersion trading exploits the wedge between index implied volatility and
  the (weighted) average implied volatility of the index's constituents. The
  key insight is that index variance decomposes into:

    Var(Index) = sum(w_i^2 * Var(i)) + sum(w_i * w_j * Cov(i,j))

  When implied correlation (embedded in index vol) exceeds realized correlation,
  index vol is "rich" relative to single-stock vol -- the classic long-dispersion
  trade sells index vol and buys single-stock vol.

  For AI semiconductors specifically:
  - NVDA, AVGO, and TSM dominate the AI chip ecosystem with high correlation
    during AI narrative shifts but significant idiosyncratic risk (earnings,
    export controls, product cycles)
  - These names carry 2-3x the vol of QQQ, but QQQ has ~8-10% weight in NVDA
    alone, creating a natural linkage
  - Historical research (Quantpedia, academic studies) shows dispersion strategies
    on S&P 500 delivered 14-27% annual returns with Sharpe ~0.35-0.40 from
    2000-2017, though with significant tail risk during correlation spikes
  - The strategy is fundamentally a SHORT CORRELATION trade: it profits when
    stocks move independently and loses when they move together (crisis periods)
  - In the semiconductor sector, correlation tends to spike around:
    (a) Fed rate decisions, (b) China/Taiwan geopolitical events,
    (c) Broad AI sentiment shifts, (d) NVDA earnings (sector bellwether)
  - Key risk: "correlation crisis" -- during sell-offs, all semis drop together
    and the short index vol leg does not offset the long single-stock vol losses
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Pull data
    term_structures, skew_data, spot_prices, smile_snapshots = pull_all_data()

    if len(term_structures) < 2:
        print("ERROR: Not enough data pulled. Check network / yfinance connectivity.")
        sys.exit(1)

    # Compute basket IV
    print("--- Computing basket IV ---")
    basket_df = compute_basket_iv(term_structures)
    print(f"  Basket data points: {len(basket_df)}")

    # Generate plots
    print("\n--- Generating plots ---")
    plot_term_structures(term_structures, basket_df)
    plot_skew_comparison(skew_data)
    plot_smile_snapshot(smile_snapshots, spot_prices, target_dte=30)

    # Print analysis
    print_analysis(term_structures, skew_data, basket_df, spot_prices)

    # Save data to CSV for further analysis
    print("\n--- Saving data ---")
    for sym, df in term_structures.items():
        path = os.path.join(OUTPUT_DIR, f"term_structure_{sym}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    if not basket_df.empty:
        path = os.path.join(OUTPUT_DIR, "term_structure_BASKET.csv")
        basket_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
