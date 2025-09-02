"""Testing statistical helper functions for the design matrix generator"""

import numbers

import numpy as np
import pytest

from semeio.fmudesign import design_distributions as dists


@pytest.mark.parametrize("seed", range(100))
def test_draw_values_pert(seed):
    rng = np.random.default_rng(seed)

    distr = dists.PERT(low=10, mode=50, high=100)
    assert not distr.sample(size=0, rng=rng).size

    values = distr.sample(size=20, rng=rng)
    assert len(values) == 20
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(10 <= value <= 100 for value in values)

    # For symmetric PERT, verify both mode and mean are at 5
    values = dists.PERT(low=0, mode=5, high=10, scale=4.0).sample(size=5000, rng=rng)
    hist, bins = np.histogram(values, bins=100)
    empirical_mode = bins[np.argmax(hist)]
    assert np.isclose(empirical_mode, 5, atol=0.55)
    assert np.isclose(values.mean(), 5, atol=0.55)


def test_sample_discrete():
    rng = np.random.default_rng()
    outcomes = ["foo", "bar.com"]
    # Test basic functionality
    values = dists.sample_discrete([",".join(outcomes)], 10, rng)
    assert all(value in outcomes for value in values)

    # Test empty case
    assert not dists.sample_discrete([",".join(outcomes)], 0, rng).size

    # Test negative numreals
    with pytest.raises(ValueError):
        dists.sample_discrete([",".join(outcomes)], -1, rng)[1]

    # Test weighted case where only bar.com should appear
    assert "foo" not in dists.sample_discrete([",".join(outcomes), "0,1"], 10, rng)

    # Test weights that don't sum to 1
    weighted_values = dists.sample_discrete([",".join(outcomes), "2,6"], 100, rng)
    # Should see roughly 25% foo and 75% bar.com
    foo_count = np.sum(weighted_values == "foo")
    assert 15 <= foo_count <= 35  # Allow some variance due to randomness


def test_draw_values():
    """Test the wrapper function for drawing values"""
    rng = np.random.default_rng()

    values = dists.draw_values("unif", [0, 1], 10, rng)
    assert len(values) == 10
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(0 <= value <= 1 for value in values)

    values = dists.draw_values("UNIF", [0, 1], 10, rng)
    assert len(values) == 10

    values = dists.draw_values("unifORM", [0, 1], 10, rng)
    assert len(values) == 10

    values = dists.draw_values("UnifORMgarbagestillworks", [0, 1], 10, rng)
    assert len(values) == 10

    with pytest.raises(ValueError):
        dists.draw_values("non-existing-distribution", [0, 10], 100, rng)

    values = dists.draw_values("NORMAL", ["0", "1"], 10, rng)
    assert len(values) == 10

    values = dists.draw_values("LOGNORMAL", [0.1, 1], 10, rng)
    assert len(values) == 10

    values = dists.draw_values("Pert", [0.1, 1, 10], 10, rng)
    assert len(values) == 10

    values = dists.draw_values("triangular", [0.1, 1, 10], 10, rng)
    assert len(values) == 10

    values = dists.draw_values("logunif", [0.1, 1], 10, rng)
    assert len(values) == 10
