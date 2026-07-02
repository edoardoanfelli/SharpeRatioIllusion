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
