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


# Q_p(x) — the universal function of Magdon-Ismail, Atiya, Pratap & Abu-Mostafa
# (2004), "On the Maximum Drawdown of a Brownian Motion", J. Appl. Prob. 41(1):
# E[MDD] = (2 sigma^2 / mu) * Q_p(x) with x = mu^2 T / (2 sigma^2), mu > 0.
# Table calibrated in-repo by Monte Carlo with Richardson extrapolation
# (12k paths, 4096 vs 16384 steps; scripts/calibrate_qp.py regenerates it).
# Checks: Q_p(1) = 0.58785 vs 0.588642 in the paper; converges to the exact
# large-x asymptote 0.25*ln(x) + 0.49088 by x ~ 75. Accuracy ~1%.
_QP_X = np.array(
    [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
     5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]
)
_QP_Y = np.array(
    [0.08427, 0.11737, 0.17852, 0.24343, 0.32354, 0.38075, 0.47182, 0.53581,
     0.58785, 0.67006, 0.71939, 0.82002, 0.93122, 1.01297, 1.08704, 1.19034,
     1.24584, 1.35824, 1.47688, 1.56650, 1.64394]
)


def _qp(x: float) -> float:
    """Q_p(x) via log-log interpolation of the table, with exact asymptotes."""
    if x <= _QP_X[0]:
        # small-x limit: Q_p(x) -> 0.5 * sqrt(pi * x)
        return 0.5 * np.sqrt(np.pi * x)
    if x >= _QP_X[-1]:
        # large-x limit (Magdon-Ismail et al., eq. for x -> inf)
        return 0.25 * np.log(x) + 0.49088
    return float(np.exp(np.interp(np.log(x), np.log(_QP_X), np.log(_QP_Y))))


def expected_max_drawdown(
    sr_true: float,
    n_periods: int,
    *,
    vol: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Theoretical expected maximum drawdown for a given true SR and horizon.

    Closed-form approximation for Brownian motion with drift (Magdon-Ismail et al.
    2004; the framework behind Bouchaud et al. 2017). Inputs mirror
    ``simulation.simulate_returns``: ``sr_true`` is annualised, ``vol`` is the
    per-period volatility, and the horizon is ``n_periods`` periods. Returns the
    expected depth as a *positive* number in the same units as the PnL.

    Continuous-time result; discrete daily sampling underestimates it slightly,
    so Monte Carlo estimates land a few percent below this value.
    """
    if n_periods <= 0:
        raise ValueError("n_periods must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")
    sigma2_t = vol**2 * n_periods
    if sr_true == 0:
        # driftless BM: E[MDD] = sqrt(pi/2) * sigma * sqrt(T)
        return float(np.sqrt(np.pi / 2) * np.sqrt(sigma2_t))
    if sr_true < 0:
        raise ValueError("expected_max_drawdown requires sr_true >= 0")
    mu = sr_true * vol / np.sqrt(periods_per_year)
    x = mu**2 * n_periods / (2 * vol**2)
    return float(2 * vol**2 / mu * _qp(x))
