"""Drawdown depth and duration under an assumed Sharpe ratio.

Following Rej, A., Seager, P. & Bouchaud, J.-P. (2017), "You are in a drawdown.
When should you start worrying?" (Capital Fund Management), the goal is to map an
*assumed* true Sharpe ratio to the distribution of drawdown depth and length, so a
running drawdown can be judged "normal" or "alarming" relative to that hypothesis.

All functions are stubs for now.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def drawdown_series(pnl: np.ndarray) -> np.ndarray:
    """Drawdown at each time step: ``pnl - running_max(pnl)`` (<= 0)."""
    raise NotImplementedError("Implemented in the next phase.")


def max_drawdown(pnl: np.ndarray) -> float:
    """Maximum drawdown depth (a non-positive number) of a PnL path."""
    raise NotImplementedError("Implemented in the next phase.")


def longest_drawdown_duration(pnl: np.ndarray) -> int:
    """Length (in periods) of the longest interval spent below a prior peak."""
    raise NotImplementedError("Implemented in the next phase.")


def expected_max_drawdown(
    sr_true: float,
    n_periods: int,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Theoretical expected maximum drawdown for a given true SR and horizon.

    For a Brownian motion with drift, expected max drawdown has closed-form
    approximations in the SR and horizon (see Bouchaud et al. 2017). This wraps
    that mapping so notebooks can compare simulated drawdowns against the
    theoretical "worry threshold".
    """
    raise NotImplementedError("Implemented in the next phase.")
