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

Convention: PSR / MinTRL / DSR operate on the *per-observation* Sharpe ratio and its
moments, all at the same frequency. ``kurtosis`` is the full (non-excess) kurtosis,
so a Gaussian has ``kurtosis = 3``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

TRADING_DAYS_PER_YEAR = 252
_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(
    returns: np.ndarray,
    *,
    rf: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Sample Sharpe ratio ``(mean(returns) - rf) / std(returns)``.

    ``rf`` is the per-period risk-free rate. If ``periods_per_year`` is given, the
    result is annualised by ``sqrt(periods_per_year)``. Uses ``ddof=1``.
    """
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1)
    if std == 0:
        return float("nan")
    sr = (returns.mean() - rf) / std
    if periods_per_year is not None:
        sr *= np.sqrt(periods_per_year)
    return float(sr)


def sharpe_std_error(
    returns: np.ndarray,
    *,
    account_for_autocorrelation: bool = False,
    max_lag: int = 10,
) -> float:
    """Standard error of the (per-period) Sharpe ratio estimator.

    Uses the four-moment expression (Mertens; Lo 2002): for a per-period estimate
    ``sr`` over ``n`` observations with skewness ``g3`` and kurtosis ``g4``,

        Var(sr) ≈ (1 - g3*sr + (g4 - 1)/4 * sr**2) / n

    which reduces to the familiar ``(1 + 0.5*sr**2)/n`` under normality
    (``g3 = 0``, ``g4 = 3``).

    When ``account_for_autocorrelation`` is True, the IID variance is scaled by a
    Newey-West style factor ``1 + 2 * sum_{k=1..K} (1 - k/(K+1)) * rho_k`` built from
    the sample autocorrelations, capturing the inflation of the annualised SR under
    serial dependence discussed in Lo (2002).
    """
    returns = np.asarray(returns, dtype=float)
    n = returns.size
    if n < 2:
        return float("nan")
    sr = sharpe_ratio(returns)
    g3 = float(stats.skew(returns, bias=False))
    g4 = float(stats.kurtosis(returns, fisher=False, bias=False))
    var = (1 - g3 * sr + (g4 - 1) / 4 * sr**2) / n

    if account_for_autocorrelation:
        var *= _newey_west_factor(returns, max_lag)

    return float(np.sqrt(max(var, 0.0)))


def _newey_west_factor(returns: np.ndarray, max_lag: int) -> float:
    """Bartlett-weighted serial-dependence inflation factor (>= 0)."""
    x = returns - returns.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 1.0
    k = min(max_lag, returns.size - 1)
    factor = 1.0
    for lag in range(1, k + 1):
        rho = np.sum(x[lag:] * x[:-lag]) / denom
        weight = 1.0 - lag / (k + 1)
        factor += 2.0 * weight * rho
    return max(factor, 0.0)


def _moment_denominator(sr_hat: float, skew: float, kurtosis: float) -> float:
    """The ``sqrt`` denominator shared by PSR and MinTRL (variance of sr_hat * n)."""
    return np.sqrt(1 - skew * sr_hat + (kurtosis - 1) / 4 * sr_hat**2)


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
    ``sr_hat`` over ``n`` observations, accounting for the first four moments::

        PSR = Phi( (sr_hat - sr_benchmark) * sqrt(n - 1)
                   / sqrt(1 - skew*sr_hat + (kurtosis - 1)/4 * sr_hat**2) )

    All Sharpe ratios are at the same (non-annualised) frequency. Returns a
    probability in ``[0, 1]``.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / _moment_denominator(
        sr_hat, skew, kurtosis
    )
    return float(stats.norm.cdf(z))


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
    SR exceeds ``sr_benchmark``::

        MinTRL = 1 + (1 - skew*sr_hat + (kurtosis - 1)/4 * sr_hat**2)
                     * (Phi_inv(confidence) / (sr_hat - sr_benchmark))**2

    ``sr_hat`` must exceed ``sr_benchmark`` (otherwise the target is unreachable and
    ``inf`` is returned). The Sharpe ratios are at the same frequency; divide the
    result by ``periods_per_year`` to read it in years.
    """
    if sr_hat <= sr_benchmark:
        return float("inf")
    denom = 1 - skew * sr_hat + (kurtosis - 1) / 4 * sr_hat**2
    z = stats.norm.ppf(confidence)
    return float(1 + denom * (z / (sr_hat - sr_benchmark)) ** 2)


def expected_max_sharpe(n_trials: int, sr_variance_across_trials: float) -> float:
    """Expected maximum of ``n_trials`` i.i.d. SR estimates (Bailey & LdP 2014).

    Extreme-value (Gumbel) approximation of the highest Sharpe ratio obtainable by
    chance when ``n_trials`` independent, truly-zero-skill strategies are tried::

        E[max SR] ≈ sqrt(V) * [ (1 - g)*Phi_inv(1 - 1/N) + g*Phi_inv(1 - 1/(N*e)) ]

    where ``V`` is the variance of the SR estimates across trials and ``g`` is the
    Euler-Mascheroni constant. This is the benchmark used by the Deflated SR.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    sigma = np.sqrt(sr_variance_across_trials)
    g = _EULER_MASCHERONI
    term = (1 - g) * stats.norm.ppf(1 - 1.0 / n_trials) + g * stats.norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    return float(sigma * term)


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

    The PSR evaluated against ``sr_benchmark = expected_max_sharpe(...)`` rather than
    zero, i.e. against the best SR expected from ``n_trials`` truly-worthless
    strategies. This corrects the reported SR for selection bias / backtest
    overfitting on top of non-normality. Returns a probability in ``[0, 1]``.
    """
    sr_benchmark = expected_max_sharpe(n_trials, sr_variance_across_trials)
    return probabilistic_sharpe_ratio(
        sr_hat,
        n,
        sr_benchmark=sr_benchmark,
        skew=skew,
        kurtosis=kurtosis,
    )
