"""Generate the figures used in the project report.

Reproducible: ``python scripts/make_figures.py`` writes PNGs into
``docs/report/figures/``. Palette and mark choices follow the repo's dataviz
guidance (fixed categorical hue order, one axis, thin marks, recessive grid,
selective direct labels).
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sharpe_illusion import drawdown, metrics, simulation  # noqa: E402

# ---- palette (validated reference instance, light surface) -----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"

PPY = simulation.TRADING_DAYS_PER_YEAR
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
        "figure.dpi": 150,
    }
)


def _title(ax, title, subtitle=None):
    ax.text(0.0, 1.14, title, transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=INK, ha="left", va="bottom")
    if subtitle:
        ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, ha="left", va="bottom")


def _save(fig, name):
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(OUT.parents[2])}")


def fig1_illusion(rng):
    """One year of daily data cannot separate SR=0 from SR=1."""
    horizon = PPY  # one year
    n_paths = 40_000
    specs = [(0.0, MUTED), (0.5, BLUE), (1.0, AQUA)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(-4, 5, 90)
    frac_over_1 = None
    for sr_true, color in specs:
        r = simulation.simulate_returns(sr_true, horizon, n_paths, rng=rng)
        srs = simulation.realized_sharpe(r)
        ax.hist(srs, bins=bins, density=True, histtype="step", lw=2, color=color)
        if sr_true == 0.0:
            frac_over_1 = float(np.mean(srs > 1.0))
    ax.axvline(1.0, color=RED, lw=1.2, ls="--")
    ax.text(1.05, ax.get_ylim()[1] * 0.92, "hire-me line (SR = 1)", color=RED,
            fontsize=9, va="top")
    # legend as direct labels
    for sr_true, color, y in [(0.0, MUTED, 0.86), (0.5, BLUE, 0.78), (1.0, AQUA, 0.70)]:
        ax.text(0.985, y, f"true SR = {sr_true:g}", transform=ax.transAxes, color=color,
                ha="right", fontsize=10, fontweight="bold")
    _title(
        ax,
        "One year of data can't tell skill from luck",
        f"Realized annual Sharpe over 252 days · {frac_over_1:.0%} of the zero-edge "
        "strategies clear the 'hire-me' line",
    )
    ax.set_xlabel("Realized (annualized) Sharpe ratio")
    ax.set_ylabel("Probability density")
    ax.set_yticks([])
    _save(fig, "fig1_illusion.png")
    return frac_over_1


def fig2_drawdown(rng):
    """A genuinely profitable strategy can stay underwater for years."""
    years = 10
    horizon = PPY * years
    n_paths = 4000
    sr_true = 0.5
    r = simulation.simulate_returns(sr_true, horizon, n_paths, rng=rng)
    pnl = simulation.cumulative_pnl(r)
    t = np.arange(1, horizon + 1) / PPY

    fig, ax = plt.subplots(figsize=(8, 4.6))
    # background cloud of sample paths
    for i in range(0, 120):
        ax.plot(t, pnl[i], color=GRID, lw=0.6, zorder=1)
    # theoretical drift
    drift = sr_true / np.sqrt(PPY) * np.arange(1, horizon + 1)
    ax.plot(t, drift, color=BLUE, lw=2.2, zorder=3, label="expected PnL")
    # highlight a genuinely-positive path that stays negative for years
    underwater_len = np.array(
        [np.argmax(pnl[i] > 0) if np.any(pnl[i] > 0) else horizon for i in range(500)]
    )
    worst = int(np.argmax(underwater_len))
    years_neg = underwater_len[worst] / PPY
    ax.plot(t, pnl[worst], color=RED, lw=2.0, zorder=4)
    ax.axhline(0, color=BASELINE, lw=1.0)
    ax.text(0.985, 0.10, "expected PnL (true SR = 0.5)", transform=ax.transAxes,
            color=BLUE, ha="right", fontsize=10, fontweight="bold")
    ax.text(0.985, 0.03, f"one unlucky path: {years_neg:.1f} years in the red",
            transform=ax.transAxes, color=RED, ha="right", fontsize=10,
            fontweight="bold")

    frac_neg_3y = float(np.mean(pnl[:, PPY * 3 - 1] < 0))
    _title(
        ax,
        "A profitable strategy can lose money for years",
        f"Same true edge (SR = 0.5), 4000 paths · {frac_neg_3y:.0%} are still in the "
        "red after 3 years",
    )
    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative PnL (in daily-vol units)")
    ax.set_xlim(0, years)
    _save(fig, "fig2_drawdown.png")
    return frac_neg_3y, years_neg


def fig3_mintrl():
    """Years of daily data needed before a Sharpe ratio is trustworthy."""
    sr_levels = [0.25, 0.5, 1.0, 2.0]
    years = []
    for sr in sr_levels:
        sr_pp = sr / np.sqrt(PPY)
        obs = metrics.minimum_track_record_length(sr_pp, confidence=0.975)
        years.append(obs / PPY)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(sr_levels))
    ax.bar(xs, years, width=0.6, color=BLUE, zorder=2)
    for x, y in zip(xs, years):
        ax.text(x, y + max(years) * 0.015, f"{y:.0f} yr", ha="center", va="bottom",
                color=INK, fontweight="bold", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"SR = {s:g}" for s in sr_levels])
    ax.set_ylabel("Years of daily data for 97.5% confidence")
    ax.set_ylim(0, max(years) * 1.15)
    _title(
        ax,
        "How long before you can trust a Sharpe ratio?",
        "Minimum track record length to be confident the true SR > 0 "
        "(Bailey & Lopez de Prado)",
    )
    ax.grid(axis="x", visible=False)
    _save(fig, "fig3_mintrl.png")
    return dict(zip(sr_levels, years))


def fig4_overfitting(rng):
    """Trying many strategies manufactures a high Sharpe ratio by luck."""
    horizon = PPY * 2  # two-year backtests
    n_configs = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    reps = 400
    expected_max = []
    for n in n_configs:
        best = np.empty(reps)
        for j in range(reps):
            r = simulation.simulate_returns(0.0, horizon, n, rng=rng)
            best[j] = np.nanmax(simulation.realized_sharpe(r))
        expected_max.append(best.mean())

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(n_configs, expected_max, color=BLUE, lw=2.2, marker="o", ms=5, zorder=3)
    ax.set_xscale("log")
    for n, e in [(n_configs[0], expected_max[0]), (n_configs[-1], expected_max[-1])]:
        ax.annotate(f"{e:.2f}", (n, e), textcoords="offset points", xytext=(0, 8),
                    ha="center", color=INK, fontweight="bold", fontsize=10)
    ax.axhline(1.0, color=RED, lw=1.2, ls="--")
    ax.text(1.1, 1.03, "hire-me line (SR = 1)", color=RED, fontsize=9)
    _title(
        ax,
        "Search hard enough and luck looks like edge",
        "Best realized SR across N zero-edge, 2-year backtests · the reason a raw "
        "Sharpe must be deflated",
    )
    ax.set_xlabel("Number of strategy variants tried (log scale)")
    ax.set_ylabel("Best realized Sharpe by pure luck")
    _save(fig, "fig4_overfitting.png")
    return dict(zip(n_configs, expected_max))


def main():
    rng = np.random.default_rng(20260702)
    f1 = fig1_illusion(rng)
    f2 = fig2_drawdown(rng)
    f3 = fig3_mintrl()
    f4 = fig4_overfitting(rng)
    print("\n--- headline numbers for the article ---")
    print(f"zero-edge paths above SR=1 in one year: {f1:.1%}")
    print(f"good strategies still in the red after 3 years: {f2[0]:.1%}")
    print(f"longest red streak in sample: {f2[1]:.1f} years")
    print(f"years to trust SR: {f3}")
    print(f"best-of-N zero-edge SR: {f4}")


if __name__ == "__main__":
    main()
