import numpy as np

from sharpe_illusion import simulation as sim


def test_simulate_returns_shape():
    rng = np.random.default_rng(0)
    r = sim.simulate_returns(0.5, n_periods=252, n_paths=7, rng=rng)
    assert r.shape == (7, 252)


def test_realized_sharpe_converges_to_true():
    # Averaged over many long paths, the realized SR should sit near sr_true.
    rng = np.random.default_rng(1)
    r = sim.simulate_returns(0.8, n_periods=252 * 40, n_paths=200, rng=rng)
    srs = sim.realized_sharpe(r)
    assert abs(np.mean(srs) - 0.8) < 0.1


def test_zero_edge_still_produces_high_sr_paths():
    # The illusion: with sr_true = 0 over ~1y, some paths look great by luck.
    rng = np.random.default_rng(2)
    r = sim.simulate_returns(0.0, n_periods=252, n_paths=5000, rng=rng)
    srs = sim.realized_sharpe(r)
    assert np.mean(srs > 1.0) > 0.02  # a non-trivial minority of false positives


def test_cumulative_pnl_matches_cumsum():
    r = np.array([[1.0, -2.0, 3.0]])
    np.testing.assert_allclose(sim.cumulative_pnl(r), [[1.0, -1.0, 2.0]])
