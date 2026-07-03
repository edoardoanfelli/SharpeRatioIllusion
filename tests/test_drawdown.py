import numpy as np

from sharpe_illusion import drawdown as dd


def test_drawdown_series_non_positive():
    pnl = np.array([0.0, 1.0, 0.5, 2.0, 1.0])
    series = dd.drawdown_series(pnl)
    assert np.all(series <= 0)
    np.testing.assert_allclose(series, [0.0, 0.0, -0.5, 0.0, -1.0])


def test_max_drawdown_value():
    pnl = np.array([0.0, 1.0, 0.5, 2.0, -1.0])
    assert dd.max_drawdown(pnl) == -3.0  # peak 2.0 down to -1.0


def test_longest_drawdown_duration():
    pnl = np.array([0.0, 1.0, 0.5, 0.4, 0.9, 2.0])  # underwater for steps 2,3,4
    assert dd.longest_drawdown_duration(pnl) == 3


def test_time_underwater_fraction():
    pnl = np.array([0.0, 1.0, 0.5, 2.0])  # underwater at 1 of 4 steps
    assert dd.time_underwater_fraction(pnl) == 0.25


def test_max_drawdown_vectorised():
    pnl = np.array([[0.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    np.testing.assert_allclose(dd.max_drawdown(pnl), [-2.0, -1.0])


def test_expected_max_drawdown_zero_drift_exact():
    # Driftless case has the exact closed form sqrt(pi/2) * sigma * sqrt(n).
    n = 252 * 4
    expected = np.sqrt(np.pi / 2) * np.sqrt(n)
    assert dd.expected_max_drawdown(0.0, n) == expected


def test_expected_max_drawdown_matches_monte_carlo():
    # Continuous-time value should sit within a few percent of daily-sampled MC
    # (discrete sampling biases the MC slightly low).
    from sharpe_illusion import simulation as sim

    rng = np.random.default_rng(11)
    n = 252 * 10
    r = sim.simulate_returns(0.5, n, n_paths=3000, rng=rng)
    pnl = sim.cumulative_pnl(r)
    mc = -np.mean(dd.max_drawdown(pnl))
    theory = dd.expected_max_drawdown(0.5, n)
    assert abs(theory - mc) / mc < 0.05


def test_expected_max_drawdown_monotonic_in_horizon():
    assert dd.expected_max_drawdown(0.5, 252 * 20) > dd.expected_max_drawdown(
        0.5, 252 * 5
    )


def test_expected_max_drawdown_rejects_negative_sr():
    import pytest

    with pytest.raises(ValueError):
        dd.expected_max_drawdown(-0.5, 252)
