"""Drawdown depth and duration under an assumed Sharpe ratio.

Following Rej, A., Seager, P. & Bouchaud, J.-P. (2017), "You are in a drawdown.
When should you start worrying?" (Capital Fund Management), the goal is to map an
*assumed* true Sharpe ratio to the distribution of drawdown depth and length, so a
running drawdown can be judged "normal" or "alarming" relative to that hypothesis.

Here PnL is treated as an additive cumulative-return series (log-PnL), so a drawdown
is measured as ``pnl - running_max(pnl)`` (a non-positive quantity).
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def drawdown_series(pnl: np.ndarray) -> np.ndarray:
    """Drawdown at each step: ``pnl - running_max(pnl)`` (<= 0), along the last axis."""
    pnl = np.asarray(pnl, dtype=float)
    running_max = np.maximum.accumulate(pnl, axis=-1)
    return pnl - running_max


def max_drawdown(pnl: np.ndarray) -> np.ndarray | float:
    """Maximum drawdown depth (a non-positive number) along the last axis.

    Returns a scalar for a 1-D path or an array (one value per path) for 2-D input.
    """
    dd = drawdown_series(pnl)
    mdd = dd.min(axis=-1)
    return float(mdd) if np.ndim(mdd) == 0 else mdd


def longest_drawdown_duration(pnl: np.ndarray) -> int:
    """Length (in periods) of the longest interval spent strictly below a prior peak.

    Expects a 1-D PnL path.
    """
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("longest_drawdown_duration expects a 1-D path")
    dd = drawdown_series(pnl)
    underwater = dd < 0
    longest = current = 0
    for below in underwater:
        current = current + 1 if below else 0
        if current > longest:
            longest = current
    return int(longest)


def time_underwater_fraction(pnl: np.ndarray) -> np.ndarray | float:
    """Fraction of periods spent below a prior peak, along the last axis."""
    dd = drawdown_series(pnl)
    frac = (dd < 0).mean(axis=-1)
    return float(frac) if np.ndim(frac) == 0 else frac


def expected_max_drawdown(
    sr_true: float,
    n_periods: int,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Theoretical expected maximum drawdown for a given true SR and horizon.

    Closed-form approximation for Brownian motion with drift (Magdon-Ismail et al.;
    Bouchaud et al. 2017). Not yet implemented — the notebooks currently rely on the
    Monte Carlo drawdown statistics above; this closed form lands in the next phase.
    """
    raise NotImplementedError("Closed-form expected max drawdown lands next phase.")
