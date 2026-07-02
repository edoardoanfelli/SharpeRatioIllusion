import numpy as np
import pytest

from sharpe_illusion import metrics as m


def test_sharpe_ratio_basic():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = (r.mean()) / r.std(ddof=1)
    assert m.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_ratio_annualised():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    assert m.sharpe_ratio(r, periods_per_year=252) == pytest.approx(
        m.sharpe_ratio(r) * np.sqrt(252)
    )


def test_std_error_reduces_to_iid_normal():
    # For near-Gaussian returns, SE ~= sqrt((1 + 0.5 sr^2)/n).
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, size=5000)
    sr = m.sharpe_ratio(r)
    iid = np.sqrt((1 + 0.5 * sr**2) / r.size)
    assert m.sharpe_std_error(r) == pytest.approx(iid, rel=0.15)


def test_psr_is_probability_and_monotonic_in_n():
    p_small = m.probabilistic_sharpe_ratio(0.1, n=50)
    p_large = m.probabilistic_sharpe_ratio(0.1, n=5000)
    assert 0.0 <= p_small <= 1.0 <= 1.0
    assert p_large > p_small  # more data -> more confident the true SR > 0


def test_negative_skew_lowers_psr():
    base = m.probabilistic_sharpe_ratio(0.15, n=250, skew=0.0, kurtosis=3.0)
    skewed = m.probabilistic_sharpe_ratio(0.15, n=250, skew=-1.0, kurtosis=6.0)
    assert skewed < base


def test_mintrl_positive_and_infinite_below_benchmark():
    assert m.minimum_track_record_length(0.1, sr_benchmark=0.0) > 1
    assert m.minimum_track_record_length(0.0, sr_benchmark=0.1) == float("inf")


def test_deflated_sr_below_psr_with_many_trials():
    # Deflating against many trials must not increase confidence.
    psr = m.probabilistic_sharpe_ratio(0.15, n=1000)
    dsr = m.deflated_sharpe_ratio(
        0.15, n=1000, n_trials=100, sr_variance_across_trials=0.01
    )
    assert dsr < psr


def test_expected_max_sharpe_grows_with_trials():
    e10 = m.expected_max_sharpe(10, 0.01)
    e1000 = m.expected_max_sharpe(1000, 0.01)
    assert 0 < e10 < e1000
