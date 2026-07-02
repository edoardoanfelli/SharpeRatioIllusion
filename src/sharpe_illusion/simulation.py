"""Monte Carlo simulation of PnL paths from a *fixed true* Sharpe ratio.

The central experiment of this repository: choose a true (population) Sharpe ratio
``sr_true`` — e.g. 0.0, 0.5 or 1.0 in annual units — generate many return paths that
have exactly that population SR, and observe how the *realized* SR and the cumulative
PnL scatter around the theoretical value over finite horizons.

This makes the "illusion" concrete:
- with ``sr_true = 0`` some paths still exhibit a high realized SR (false positives);
- with ``sr_true > 0`` some paths stay in drawdown for years (false negatives).
"""

from __future__ import annotations

import numpy as np

# A trading year is conventionally 252 business days.
TRADING_DAYS_PER_YEAR = 252


def simulate_returns(
    sr_true: float,
    n_periods: int,
    n_paths: int = 1,
    *,
    vol: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate i.i.d. Gaussian return paths with a given *annualised* true SR.

    The per-period mean is chosen so that
    ``mean / vol * sqrt(periods_per_year) == sr_true`` at the population level.

    Parameters
    ----------
    sr_true:
        Annualised true Sharpe ratio of the data-generating process.
    n_periods:
        Number of periods (e.g. days) per path.
    n_paths:
        Number of independent Monte Carlo paths.
    vol:
        Per-period return standard deviation.
    periods_per_year:
        Periods used to annualise (252 for daily data).
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_periods)`` of per-period returns.
    """
    if n_periods <= 0:
        raise ValueError("n_periods must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")

    rng = np.random.default_rng() if rng is None else rng
    per_period_mean = sr_true * vol / np.sqrt(periods_per_year)
    return rng.normal(loc=per_period_mean, scale=vol, size=(n_paths, n_periods))


def cumulative_pnl(returns: np.ndarray) -> np.ndarray:
    """Cumulative (summed) PnL along the last axis of a returns array."""
    return np.cumsum(np.asarray(returns, dtype=float), axis=-1)


def realized_sharpe(
    returns: np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Annualised realized (sample) Sharpe ratio per path.

    Operates along the last axis, so a ``(n_paths, n_periods)`` array returns one
    Sharpe ratio per path. Uses the sample standard deviation (``ddof=1``).

    Note: this is the *naive* estimator; see :mod:`sharpe_illusion.metrics` for its
    standard error and the bias / selection corrections.
    """
    returns = np.asarray(returns, dtype=float)
    mean = returns.mean(axis=-1)
    std = returns.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std, np.nan)
    return sr * np.sqrt(periods_per_year)
