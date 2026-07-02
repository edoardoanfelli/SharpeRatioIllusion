"""Scaffolding smoke tests.

These confirm the package is importable and its public API is wired up. The
analytical functions are stubs at this stage and are expected to raise
NotImplementedError; that expectation is asserted here so the tests turn green
once the bodies are implemented and the raises removed.
"""

import numpy as np
import pytest

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


def test_stubs_raise_until_implemented():
    with pytest.raises(NotImplementedError):
        simulation.simulate_returns(sr_true=0.5, n_periods=252)
    with pytest.raises(NotImplementedError):
        metrics.sharpe_ratio(np.zeros(10))
