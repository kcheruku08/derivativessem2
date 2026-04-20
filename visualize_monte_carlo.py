"""
Monte Carlo Visualization
==========================
Generates plots from the Monte Carlo simulation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_output")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "monte_carlo_results.csv")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "grid.alpha": 0.3,
})


def load_results():
    df = pd.read_csv(RESULTS_PATH)
    return df


def plot_sharpe_distribution(df):
    """Histogram of Sharpe ratios: strategy vs what random would produce."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sharpes = df["sharpe"].values
    mean_s = sharpes.mean()
    median_s = np.median(sharpes)

    ax.hist(sharpes, bins=50, alpha=0.85, color="#2196F3", edgecolor="white",
            linewidth=0.5, label=f"Strategy (n={len(sharpes)})")

    ax.axvline(mean_s, color="#E91E63", linewidth=2.5, linestyle="-",
               label=f"Mean: {mean_s:.2f}")
    ax.axvline(median_s, color="#FF9800", linewidth=2, linestyle="--",
               label=f"Median: {median_s:.2f}")
    ax.axvline(0, color="black", linewidth=1.5, linestyle=":", alpha=0.7,
               label="Sharpe = 0 (no edge)")
    ax.axvline(0.5, color="#4CAF50", linewidth=1.5, linestyle="-.",
               label="Sharpe = 0.5 (decent)")

    pct_above_zero = (sharpes > 0).mean() * 100
    pct_above_05 = (sharpes > 0.5).mean() * 100
    ax.text(0.97, 0.95, f"P(Sharpe > 0) = {pct_above_zero:.1f}%\nP(Sharpe > 0.5) = {pct_above_05:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: Distribution of Strategy Sharpe Ratios (1000 simulations)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)

    path = os.path.join(OUTPUT_DIR, "mc_sharpe_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_return_distribution(df):
    """Histogram of total returns with percentile markers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    returns = df["total_return"].values * 100  # percent

    ax.hist(returns, bins=60, alpha=0.85, color="#9C27B0", edgecolor="white", linewidth=0.5)

    p5 = np.percentile(returns, 5)
    p50 = np.median(returns)
    p95 = np.percentile(returns, 95)

    ax.axvline(p5, color="#F44336", linewidth=2, linestyle="--", label=f"5th pctile: {p5:.0f}%")
    ax.axvline(p50, color="#FF9800", linewidth=2.5, linestyle="-", label=f"Median: {p50:.0f}%")
    ax.axvline(p95, color="#4CAF50", linewidth=2, linestyle="--", label=f"95th pctile: {p95:.0f}%")
    ax.axvline(0, color="black", linewidth=1.5, linestyle=":", alpha=0.7)

    prob_profit = (returns > 0).mean() * 100
    ax.text(0.97, 0.95, f"P(profitable) = {prob_profit:.1f}%\nMean: {returns.mean():.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: Distribution of 5-Year Total Returns")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)

    path = os.path.join(OUTPUT_DIR, "mc_return_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_drawdown_distribution(df):
    """Max drawdown distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))

    dd = df["max_dd"].values * 100  # percent (negative values)

    ax.hist(dd, bins=50, alpha=0.85, color="#F44336", edgecolor="white", linewidth=0.5)

    p5 = np.percentile(dd, 5)
    p50 = np.median(dd)
    mean_dd = dd.mean()

    ax.axvline(p5, color="#B71C1C", linewidth=2, linestyle="--", label=f"5th pctile (worst): {p5:.1f}%")
    ax.axvline(p50, color="#FF9800", linewidth=2.5, linestyle="-", label=f"Median: {p50:.1f}%")
    ax.axvline(mean_dd, color="#2196F3", linewidth=2, linestyle="-.", label=f"Mean: {mean_dd:.1f}%")

    ax.text(0.03, 0.95, f"Worst DD: {dd.min():.1f}%\n95% of sims: DD > {p5:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: Distribution of Maximum Drawdowns")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)

    path = os.path.join(OUTPUT_DIR, "mc_drawdown_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sharpe_vs_trades(df):
    """Scatter: Sharpe vs number of trades."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["n_trades"], df["sharpe"], alpha=0.4, s=20, c=df["win_rate"],
               cmap="RdYlGn", edgecolors="none")

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Win Rate")

    ax.axhline(0, color="black", linewidth=1, linestyle=":", alpha=0.5)
    ax.axhline(df["sharpe"].mean(), color="#E91E63", linewidth=2, linestyle="-",
               label=f"Mean Sharpe: {df['sharpe'].mean():.2f}")

    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs Trade Count (colored by Win Rate)")
    ax.legend()
    ax.grid(True)

    path = os.path.join(OUTPUT_DIR, "mc_sharpe_vs_trades.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_win_rate_vs_profit_factor(df):
    """Scatter: Win rate vs profit factor."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pf = np.clip(df["profit_factor"].values, 0, 15)
    wr = df["win_rate"].values * 100

    scatter = ax.scatter(wr, pf, alpha=0.4, s=25, c=df["sharpe"],
                         cmap="coolwarm", edgecolors="none", vmin=-0.5, vmax=2.0)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio")

    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", alpha=0.7, label="PF = 1 (breakeven)")

    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Profit Factor")
    ax.set_title("Win Rate vs Profit Factor (colored by Sharpe)")
    ax.legend()
    ax.grid(True)

    path = os.path.join(OUTPUT_DIR, "mc_winrate_vs_pf.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_dashboard(df):
    """Single-page summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("COMPLACENCY FADE STRATEGY — MONTE CARLO SUMMARY (1000 simulations)",
                 fontsize=15, fontweight="bold", y=0.98)

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Sharpe histogram
    ax1 = fig.add_subplot(gs[0, 0])
    sharpes = df["sharpe"].values
    ax1.hist(sharpes, bins=40, alpha=0.85, color="#2196F3", edgecolor="white", linewidth=0.3)
    ax1.axvline(sharpes.mean(), color="#E91E63", linewidth=2)
    ax1.axvline(0, color="black", linewidth=1, linestyle=":")
    ax1.set_title(f"Sharpe Distribution\nMean: {sharpes.mean():.2f}")
    ax1.set_xlabel("Sharpe")
    ax1.grid(True)

    # 2. Return histogram
    ax2 = fig.add_subplot(gs[0, 1])
    rets = df["total_return"].values * 100
    ax2.hist(rets, bins=40, alpha=0.85, color="#9C27B0", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=1, linestyle=":")
    ax2.set_title(f"Total Return Distribution\nMedian: {np.median(rets):.0f}%")
    ax2.set_xlabel("Return (%)")
    ax2.grid(True)

    # 3. Max DD histogram
    ax3 = fig.add_subplot(gs[0, 2])
    dd = df["max_dd"].values * 100
    ax3.hist(dd, bins=40, alpha=0.85, color="#F44336", edgecolor="white", linewidth=0.3)
    ax3.set_title(f"Max Drawdown\nMedian: {np.median(dd):.1f}%")
    ax3.set_xlabel("Max DD (%)")
    ax3.grid(True)

    # 4. Sharpe vs trades scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df["n_trades"], df["sharpe"], alpha=0.3, s=12, c="#2196F3")
    ax4.axhline(sharpes.mean(), color="#E91E63", linewidth=1.5, linestyle="-")
    ax4.set_xlabel("Trades")
    ax4.set_ylabel("Sharpe")
    ax4.set_title("Sharpe vs Trade Count")
    ax4.grid(True)

    # 5. Win rate histogram
    ax5 = fig.add_subplot(gs[1, 1])
    wr = df["win_rate"].values * 100
    ax5.hist(wr, bins=30, alpha=0.85, color="#4CAF50", edgecolor="white", linewidth=0.3)
    ax5.axvline(50, color="black", linewidth=1, linestyle=":")
    ax5.set_title(f"Win Rate Distribution\nMean: {wr.mean():.1f}%")
    ax5.set_xlabel("Win Rate (%)")
    ax5.grid(True)

    # 6. Key stats text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    stats_text = (
        f"STATISTICAL SIGNIFICANCE\n"
        f"{'─' * 30}\n"
        f"P(profitable):      {(df['total_return'] > 0).mean():.1%}\n"
        f"P(Sharpe > 0):      {(df['sharpe'] > 0).mean():.1%}\n"
        f"P(Sharpe > 0.5):    {(df['sharpe'] > 0.5).mean():.1%}\n"
        f"P(Sharpe > 1.0):    {(df['sharpe'] > 1.0).mean():.1%}\n"
        f"\n"
        f"EXPECTED OUTCOMES\n"
        f"{'─' * 30}\n"
        f"E[Sharpe]:          {sharpes.mean():.2f}\n"
        f"E[Return]:          {rets.mean():.0f}%\n"
        f"E[Max DD]:          {dd.mean():.1f}%\n"
        f"E[Trades]:          {df['n_trades'].mean():.0f}\n"
        f"E[Win Rate]:        {wr.mean():.1f}%\n"
        f"\n"
        f"WORST CASE (1st pctile)\n"
        f"{'─' * 30}\n"
        f"Return:             {np.percentile(rets, 1):.0f}%\n"
        f"Max DD:             {np.percentile(dd, 1):.1f}%\n"
        f"Sharpe:             {np.percentile(sharpes, 1):.2f}\n"
    )

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, fontfamily="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))

    path = os.path.join(OUTPUT_DIR, "mc_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_equity_fan_chart(mc_config_n_days=1255):
    """
    Generate a few synthetic paths and plot equity curves to show the fan.
    Uses a subset of the MC results to show the range of outcomes.
    """
    df = load_results()
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort by total return and pick percentile representative paths
    sorted_idx = df["total_return"].argsort()
    percentiles = [5, 25, 50, 75, 95]
    colors = ["#F44336", "#FF9800", "#2196F3", "#4CAF50", "#1B5E20"]

    # Since we don't have full equity paths stored, simulate the expected
    # equity path shapes using final returns + assumed compound growth
    n_days = mc_config_n_days
    days = np.arange(n_days)

    for pct, color in zip(percentiles, colors):
        idx = sorted_idx[int(pct / 100 * len(sorted_idx))]
        total_ret = df["total_return"].iloc[idx]
        daily_ret = (1 + total_ret) ** (1/n_days) - 1

        # Add some realistic noise around the growth path
        rng = np.random.default_rng(idx)
        noise = rng.normal(0, 0.005, n_days)
        equity = np.cumprod(1 + daily_ret + noise)
        equity = equity / equity[-1] * (1 + total_ret)  # scale to match final

        ax.plot(days / 252, equity, color=color, linewidth=1.5, alpha=0.8,
                label=f"{pct}th pctile (return: {total_ret:.0%})")

    # Shade between 5th and 95th
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Years")
    ax.set_ylabel("Equity (starting at $1)")
    ax.set_title("Monte Carlo: Representative Equity Curves by Percentile")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_xlim(0, n_days / 252)

    path = os.path.join(OUTPUT_DIR, "mc_equity_fan.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  GENERATING MONTE CARLO VISUALIZATIONS")
    print("=" * 60)

    df = load_results()
    print(f"  Loaded {len(df)} simulation results\n")

    plot_sharpe_distribution(df)
    plot_return_distribution(df)
    plot_drawdown_distribution(df)
    plot_sharpe_vs_trades(df)
    plot_win_rate_vs_profit_factor(df)
    plot_summary_dashboard(df)
    plot_equity_fan_chart()

    print("\n  All visualizations saved to backtest_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
