"""Sharpe ratio estimators and their corrections.

References
----------
Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*
    58(4). — point estimate and standard error, IID and autocorrelated cases.
Bailey, D. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
    *Journal of Risk* 15(2). — Probabilistic Sharpe Ratio (PSR) and Minimum Track
    Record Length (MinTRL).
Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio
    Management* 40(5). — Deflated Sharpe Ratio (DSR).

All functions are stubs for now — signatures and formulas are fixed in the
docstrings; numerical bodies land in the next phase.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: np.ndarray,
    *,
    rf: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Sample Sharpe ratio ``(mean(returns) - rf) / std(returns)``.

    If ``periods_per_year`` is given, the result is annualised by ``sqrt(...)``.
    """
    raise NotImplementedError("Implemented in the next phase.")


def sharpe_std_error(
    returns: np.ndarray,
    *,
    account_for_autocorrelation: bool = False,
) -> float:
    """Standard error of the Sharpe ratio estimator.

    Under IID normal returns (Lo 2002), for a per-period SR estimate ``sr`` over
    ``n`` observations:

        SE(sr) ≈ sqrt( (1 + 0.5 * sr**2) / n )

    A more general expression corrects for skewness ``g3`` and kurtosis ``g4``:

        SE(sr) ≈ sqrt( (1 - g3*sr + (g4 - 1)/4 * sr**2) / n )

    When ``account_for_autocorrelation`` is True, an autocorrelation-adjusted
    variance (Lo 2002, eq. for the ``q``-period case) is used instead.
    """
    raise NotImplementedError("Implemented in the next phase.")


def probabilistic_sharpe_ratio(
    sr_hat: float,
    n: int,
    *,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).

    Probability that the *true* SR exceeds ``sr_benchmark`` given the estimate
    ``sr_hat`` over ``n`` observations, accounting for the first four moments:

        PSR = Phi( (sr_hat - sr_benchmark) * sqrt(n - 1)
                   / sqrt(1 - skew*sr_hat + (kurtosis - 1)/4 * sr_hat**2) )

    where ``Phi`` is the standard normal CDF. All quantities are in the *same*
    (non-annualised) frequency.
    """
    raise NotImplementedError("Implemented in the next phase.")


def minimum_track_record_length(
    sr_hat: float,
    *,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado 2012).

    Number of observations required for PSR to reach ``confidence`` that the true
    SR exceeds ``sr_benchmark``:

        MinTRL = 1 + (1 - skew*sr_hat + (kurtosis - 1)/4 * sr_hat**2)
                     * (Phi_inv(confidence) / (sr_hat - sr_benchmark))**2
    """
    raise NotImplementedError("Implemented in the next phase.")


def deflated_sharpe_ratio(
    sr_hat: float,
    n: int,
    *,
    n_trials: int,
    sr_variance_across_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    The PSR evaluated against a benchmark that is *not* zero but the expected
    maximum Sharpe ratio obtainable by chance across ``n_trials`` independent
    strategy configurations. The benchmark uses the variance of the SR estimates
    across trials and an extreme-value (Gumbel) approximation of the expected
    maximum. This is the correction for selection bias / backtest overfitting.
    """
    raise NotImplementedError("Implemented in the next phase.")
