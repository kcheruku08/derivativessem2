"""
Real Backtest Visualization
============================
Plots from the actual historical backtest equity curve and trade log.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_output")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "grid.alpha": 0.3,
})


def plot_equity_curve():
    """Equity curve with drawdown and position shading."""
    eq = pd.read_csv(os.path.join(OUTPUT_DIR, "regime_equity_curve.csv"), parse_dates=["Date"])
    trades = pd.read_csv(os.path.join(OUTPUT_DIR, "regime_trade_log.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={"height_ratios": [4, 1.5, 1]},
                                         sharex=True)

    # -- Equity curve --
    dates = eq["Date"]
    equity = eq["Equity"].values

    ax1.plot(dates, equity, color="#2196F3", linewidth=1.5, label="Strategy Equity")
    ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax1.fill_between(dates, 1.0, equity, where=equity >= 1.0, alpha=0.1, color="#4CAF50")
    ax1.fill_between(dates, 1.0, equity, where=equity < 1.0, alpha=0.1, color="#F44336")

    # Mark trades
    if "Entry Date" in trades.columns and not trades.empty:
        for _, t in trades.iterrows():
            try:
                entry = pd.to_datetime(t["Entry Date"])
                pnl = t["Total P&L"]
                color = "#4CAF50" if pnl > 0 else "#F44336"
                idx_entry = (dates - entry).abs().idxmin()
                ax1.axvline(entry, color=color, alpha=0.3, linewidth=1)
                ax1.scatter([entry], [equity[idx_entry]], color=color, s=30, zorder=5)
            except Exception:
                continue

    ax1.set_ylabel("Equity ($1 start)")
    ax1.set_title("Complacency Fade Strategy — Historical Backtest (SMH Universe)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # -- Drawdown --
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100

    ax2.fill_between(dates, 0, drawdown, color="#F44336", alpha=0.6)
    ax2.plot(dates, drawdown, color="#B71C1C", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(min(drawdown) * 1.2, 2)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True)

    # -- Position indicator --
    pos = eq["Position"].values
    colors = ["#4CAF50" if p == "SHORT_COMPLACENT" else "#EEEEEE" for p in pos]
    for i in range(len(dates) - 1):
        if pos[i] == "SHORT_COMPLACENT":
            ax3.axvspan(dates.iloc[i], dates.iloc[i+1], color="#4CAF50", alpha=0.5)

    ax3.set_yticks([0.5])
    ax3.set_yticklabels(["In Trade"])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Date")
    ax3.grid(True)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    path = os.path.join(OUTPUT_DIR, "backtest_equity_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_trade_pnl_waterfall():
    """Waterfall chart of individual trade P&Ls."""
    trades = pd.read_csv(os.path.join(OUTPUT_DIR, "regime_trade_log.csv"))
    if trades.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # -- Waterfall --
    pnls = trades["Total P&L"].values
    labels = [f"#{i+1}" for i in range(len(pnls))]
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    cumulative = np.cumsum(pnls)

    ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.plot(range(len(pnls)), cumulative, color="#2196F3", linewidth=2, marker="o",
             markersize=5, label="Cumulative P&L")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(range(len(pnls)))
    ax1.set_xticklabels(labels, rotation=45, fontsize=9)
    ax1.set_ylabel("P&L (per unit)")
    ax1.set_title("Trade-by-Trade P&L Waterfall")
    ax1.legend()
    ax1.grid(True, axis="y")

    # -- P&L Attribution stacked bar --
    attr_cols = [c for c in ["P&L Underlying", "P&L Gamma", "P&L Vol", "P&L Theta", "Cost Slippage"]
                 if c in trades.columns]

    if not attr_cols:
        attr_cols = [c for c in ["P&L Underlying", "P&L Gamma", "P&L Vol Move", "P&L Theta", "Cost Slippage"]
                     if c in trades.columns]

    if attr_cols:
        x = range(len(trades))
        bottom_pos = np.zeros(len(trades))
        bottom_neg = np.zeros(len(trades))

        color_map = {
            "P&L Underlying": "#2196F3",
            "P&L Gamma": "#4CAF50",
            "P&L Vol": "#FF9800",
            "P&L Vol Move": "#FF9800",
            "P&L Theta": "#9C27B0",
            "Cost Slippage": "#F44336",
        }

        for col in attr_cols:
            vals = trades[col].values
            if col == "Cost Slippage":
                vals = -vals  # show as negative
            pos_vals = np.where(vals > 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)

            label = col.replace("P&L ", "")
            c = color_map.get(col, "gray")

            ax2.bar(x, pos_vals, bottom=bottom_pos, color=c, alpha=0.8, label=label, width=0.7)
            ax2.bar(x, neg_vals, bottom=bottom_neg, color=c, alpha=0.8, width=0.7)

            bottom_pos += pos_vals
            bottom_neg += neg_vals

        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(range(len(trades)))
        ax2.set_xticklabels(labels, rotation=45, fontsize=9)
        ax2.set_ylabel("P&L Component")
        ax2.set_title("P&L Attribution by Trade")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, axis="y")

    path = os.path.join(OUTPUT_DIR, "backtest_trade_waterfall.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_monthly_returns_heatmap():
    """Monthly returns heatmap."""
    eq = pd.read_csv(os.path.join(OUTPUT_DIR, "regime_equity_curve.csv"), parse_dates=["Date"])
    eq = eq.set_index("Date")

    monthly = eq["Daily Return"].resample("ME").sum() * 100  # to percent

    years = sorted(monthly.index.year.unique())
    months = range(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            mask = (monthly.index.year == year) & (monthly.index.month == month)
            if mask.any():
                data[i, j] = monthly[mask].values[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                color = "black" if abs(val) < vmax * 0.5 else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap")

    path = os.path.join(OUTPUT_DIR, "backtest_monthly_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  GENERATING BACKTEST VISUALIZATIONS")
    print("=" * 60)

    plot_equity_curve()
    plot_trade_pnl_waterfall()
    plot_monthly_returns_heatmap()

    print("\n  All visualizations saved to backtest_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
