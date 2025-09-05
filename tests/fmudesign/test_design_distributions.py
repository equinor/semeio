"""Testing statistical helper functions for the design matrix generator"""

import numbers

import numpy as np
import pytest

from semeio.fmudesign import design_distributions as dists


def test_parameter_validation():
    params = dists.validate_params("normal", ["3.4", "-3", "1e2"])
    np.testing.assert_allclose(params, np.array([3.4, -3, 100]))

    with pytest.raises(ValueError, match="not convertible to number"):
        dists.validate_params("normal", ["3.4", "cat"])

    with pytest.raises(ValueError, match="not finite"):
        dists.validate_params("normal", ["3.4", "inf"])


def test_distribution_failures():
    # Normal distribution
    with pytest.raises(ValueError, match="Must have non-negative stddev"):
        dists.Normal(mean=0, stddev=-1)
    with pytest.raises(ValueError, match="Must have high > low"):
        dists.Normal(mean=0, stddev=1, low=5, high=4)

    # Lognormal distribution
    with pytest.raises(ValueError, match="Must have positive sigma"):
        dists.Lognormal(mean=0, sigma=-1)
    with pytest.raises(ValueError, match="Must have positive sigma"):
        dists.Lognormal(mean=0, sigma=0)

    # Uniform distribution
    with pytest.raises(ValueError, match="Must have high > low"):
        dists.Uniform(low=0, high=-5)
    with pytest.raises(ValueError, match="Must have high > low"):
        dists.Uniform(low=3, high=3)

    # Loguniform distribution
    with pytest.raises(ValueError, match="Must have 0 < low < high"):
        dists.Loguniform(low=0, high=1)
    with pytest.raises(ValueError, match="Must have 0 < low < high"):
        dists.Loguniform(low=-2, high=-1)

    # Triangular distribution
    with pytest.raises(ValueError, match="Must have low <= mode <= high"):
        dists.Triangular(low=0, mode=-1, high=5)
    with pytest.raises(ValueError, match="Must have high > low"):
        dists.Triangular(low=0, mode=7, high=-3)

    # PERT
    with pytest.raises(ValueError, match="Must have scale > 0"):
        dists.PERT(low=0, mode=5, high=10, scale=0)
    with pytest.raises(ValueError, match="Must have high > low"):
        dists.PERT(low=0, mode=5, high=-10, scale=0)


@pytest.mark.parametrize("seed", range(100))
def test_draw_values_pert(seed):
    rng = np.random.default_rng(seed)

    distr = dists.PERT(low=10, mode=50, high=100)
    assert not distr.sample(quantiles=np.array([])).size

    values = distr.sample(quantiles=rng.uniform(size=20))
    assert len(values) == 20
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(10 <= value <= 100 for value in values)

    # For symmetric PERT, verify both mode and mean are at 5
    samples = dists.generate_stratified_samples(numreals=5000, rng=rng)

    values = dists.PERT(low=0, mode=5, high=10, scale=4.0).sample(samples)
    hist, bins = np.histogram(values, bins=100)
    empirical_mode = bins[np.argmax(hist)]
    assert np.isclose(empirical_mode, 5, atol=0.55)
    assert np.isclose(values.mean(), 5, atol=0.55)


def test_sample_discrete():
    rng = np.random.default_rng()

    outcomes = ["foo", "bar.com"]
    # Test basic functionality
    values = dists.sample_discrete([",".join(outcomes)], rng.uniform(size=10))
    assert all(value in outcomes for value in values)

    # Test empty case
    assert not dists.sample_discrete([",".join(outcomes)], np.array([])).size

    # Test weighted case where only bar.com should appear
    assert "foo" not in dists.sample_discrete(
        [",".join(outcomes), "0,1"], rng.uniform(size=10)
    )

    # Test weights that don't sum to 1
    weighted_values = dists.sample_discrete(
        [",".join(outcomes), "2,6"], rng.uniform(size=100)
    )
    # Should see roughly 25% foo and 75% bar.com
    foo_count = np.sum(weighted_values == "foo")
    assert 15 <= foo_count <= 35  # Allow some variance due to randomness


def test_draw_values():
    """Test the wrapper function for drawing values"""
    rng = np.random.default_rng()

    quantiles = rng.uniform(size=10)

    values = dists.draw_values("unif", [0, 1], quantiles)
    assert len(values) == 10
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(0 <= value <= 1 for value in values)

    values = dists.draw_values("UNIF", [0, 1], quantiles)
    assert len(values) == 10

    values = dists.draw_values("unifORM", [0, 1], quantiles)
    assert len(values) == 10

    values = dists.draw_values("UnifORMgarbagestillworks", [0, 1], quantiles)
    assert len(values) == 10

    with pytest.raises(ValueError):
        dists.draw_values("non-existing-distribution", [0, 10], quantiles)

    values = dists.draw_values("NORMAL", ["0", "1"], quantiles)
    assert len(values) == 10

    values = dists.draw_values("LOGNORMAL", [0.1, 1], quantiles)
    assert len(values) == 10

    values = dists.draw_values("Pert", [0.1, 1, 10], quantiles)
    assert len(values) == 10

    values = dists.draw_values("triangular", [0.1, 1, 10], quantiles)
    assert len(values) == 10

    values = dists.draw_values("logunif", [0.1, 1], quantiles)
    assert len(values) == 10
