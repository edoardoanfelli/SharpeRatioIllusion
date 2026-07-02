"""Monte Carlo simulation of PnL paths from a *fixed true* Sharpe ratio.

The central experiment of this repository: choose a true (population) Sharpe ratio
``sr_true`` — e.g. 0.0, 0.5 or 1.0 in annual units — generate many return paths that
have exactly that population SR, and observe how the *realized* SR and the cumulative
PnL scatter around the theoretical value over finite horizons.

This makes the "illusion" concrete:
- with ``sr_true = 0`` some paths still exhibit a high realized SR (false positives);
- with ``sr_true > 0`` some paths stay in drawdown for years (false negatives).

All functions are stubs for now — signatures and references are fixed; the numerical
bodies are implemented in the next phase.
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

    The per-period mean is set so that ``mean / std * sqrt(periods_per_year)``
    equals ``sr_true`` exactly at the population level.

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
    raise NotImplementedError("Implemented in the next phase.")


def cumulative_pnl(returns: np.ndarray) -> np.ndarray:
    """Cumulative (summed) PnL along the last axis of a returns array."""
    raise NotImplementedError("Implemented in the next phase.")


def realized_sharpe(
    returns: np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Annualised realized (sample) Sharpe ratio per path.

    Note: this is the *naive* estimator; see ``metrics`` for its standard error
    and the bias/selection corrections.
    """
    raise NotImplementedError("Implemented in the next phase.")
