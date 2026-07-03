"""Smoke tests: package wiring and public API."""

import sharpe_illusion
from sharpe_illusion import drawdown, metrics, simulation


def test_package_metadata():
    assert sharpe_illusion.__version__ == "0.1.0"
    assert set(sharpe_illusion.__all__) == {"simulation", "metrics", "drawdown"}


def test_public_callables_exist():
    for fn in (
        simulation.simulate_returns,
        simulation.realized_sharpe,
        metrics.sharpe_ratio,
        metrics.probabilistic_sharpe_ratio,
        metrics.deflated_sharpe_ratio,
        metrics.minimum_track_record_length,
        drawdown.max_drawdown,
    ):
        assert callable(fn)
