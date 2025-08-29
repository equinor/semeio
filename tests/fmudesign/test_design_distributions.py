"""Testing statistical helper functions for the design matrix generator"""

import numbers

import numpy as np
import pytest

from semeio.fmudesign import design_distributions as dists

# pylint: disable=protected-access


class TestNormalDistribution:
    def test_that_empty_list_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Normal distribution must have 2 parameters or 4 for a truncated normal, but had 0 parameters.",
        ):
            dists.parse_and_validate_normal_params([])

    def test_that_empty_tuple_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Normal distribution must have 2 parameters or 4 for a truncated normal, but had 0 parameters.",
        ):
            dists.parse_and_validate_normal_params(())

    def test_that_single_parameter_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Normal distribution must have 2 parameters or 4 for a truncated normal, but had 1 parameters.",
        ):
            dists.parse_and_validate_normal_params([0])

    def test_that_three_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Normal distribution must have 2 parameters or 4 for a truncated normal, but had 3 parameters.",
        ):
            dists.parse_and_validate_normal_params([0, 0, 0])

    def test_that_five_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Normal distribution must have 2 parameters or 4 for a truncated normal, but had 5 parameters.",
        ):
            dists.parse_and_validate_normal_params([0, 0, 0, 0, 0])

    def test_that_non_numeric_parameters_raise_conversion_error(self):
        with pytest.raises(
            ValueError,
            match=r"All parameters must be convertible to numbers. Got: \['mean', 'mu'\]",
        ):
            dists.parse_and_validate_normal_params(["mean", "mu"])

    def test_that_valid_normal_parameters_return_float_tuple(self):
        assert dists.parse_and_validate_normal_params([0, 1]) == (0.0, 1.0)

    def test_that_negative_stddev_raises_validation_error(self):
        with pytest.raises(
            ValueError, match="Stddev for normal distribution must be >= 0. Got: -1.0"
        ):
            dists.parse_and_validate_normal_params([0, -1])

    def test_that_zero_stddev_is_accepted(self):
        assert dists.parse_and_validate_normal_params([0, 0]) == (0.0, 0.0)

    def test_that_valid_truncated_normal_parameters_return_four_tuple(self):
        assert dists.parse_and_validate_normal_params([0, 1, 0, 1]) == (
            0.0,
            1.0,
            0.0,
            1.0,
        )

    def test_that_invalid_truncation_bounds_raise_ordering_error(self):
        with pytest.raises(
            ValueError,
            match=r"For truncated normal distribution, lower bound must be less than upper bound, but got \[1.0, 0.0\].",
        ):
            dists.parse_and_validate_normal_params([0, 1, 1, 0])


class TestLognormalDistribution:
    def test_that_empty_list_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Lognormal distribution must have 2 parameters, but had 0 parameters.",
        ):
            dists.parse_and_validate_lognormal_params([])

    def test_that_empty_tuple_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Lognormal distribution must have 2 parameters, but had 0 parameters.",
        ):
            dists.parse_and_validate_lognormal_params(())

    def test_that_single_parameter_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Lognormal distribution must have 2 parameters, but had 1 parameters.",
        ):
            dists.parse_and_validate_lognormal_params([0])

    def test_that_three_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Lognormal distribution must have 2 parameters, but had 3 parameters.",
        ):
            dists.parse_and_validate_lognormal_params([0, 0, 0])

    def test_that_non_numeric_parameters_raise_conversion_error(self):
        with pytest.raises(
            ValueError,
            match=r"All parameters must be convertible to numbers. Got: \['mean', 'mu'\]",
        ):
            dists.parse_and_validate_lognormal_params(["mean", "mu"])

    def test_that_valid_lognormal_parameters_return_float_tuple(self):
        assert dists.parse_and_validate_lognormal_params([0, 1]) == (0.0, 1.0)

    def test_that_negative_stddev_raises_validation_error(self):
        with pytest.raises(
            ValueError,
            match="Stddev for lognormal distribution must be >= 0. Got: -1.0",
        ):
            dists.parse_and_validate_lognormal_params([0, -1])

    def test_that_zero_stddev_is_accepted(self):
        assert dists.parse_and_validate_lognormal_params([0, 0]) == (0.0, 0.0)

    def test_that_nan_parameter_raises_error(self):
        with pytest.raises(ValueError, match="Parameters cannot be NaN"):
            dists.parse_and_validate_lognormal_params([0, np.nan])

    def test_that_string_numbers_are_converted_to_floats(self):
        assert dists.parse_and_validate_lognormal_params(["0", "1"]) == (0.0, 1.0)


class TestUniformDistribution:
    def test_that_empty_list_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Uniform distribution requires exactly 2 parameters, got 0",
        ):
            dists.parse_and_validate_uniform_params([])

    def test_that_empty_tuple_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Uniform distribution requires exactly 2 parameters, got 0",
        ):
            dists.parse_and_validate_uniform_params(())

    def test_that_insufficient_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Uniform distribution requires exactly 2 parameters, got 1",
        ):
            dists.parse_and_validate_uniform_params([0])

    def test_that_excess_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Uniform distribution requires exactly 2 parameters, got 3",
        ):
            dists.parse_and_validate_uniform_params([0, 0, 0])

    def test_that_non_numeric_parameters_raise_conversion_error(self):
        with pytest.raises(ValueError, match="must be convertible to numbers"):
            dists.parse_and_validate_uniform_params(["mean", "mu"])

    def test_that_invalid_ordering_raises_constraint_error(self):
        with pytest.raises(ValueError, match="must satisfy low < high"):
            dists.parse_and_validate_uniform_params([0, -1])

    def test_that_valid_parameters_return_float_tuple(self):
        assert dists.parse_and_validate_uniform_params([0, 1]) == (0.0, 1.0)

    def test_that_equal_bounds_are_accepted(self):
        assert dists.parse_and_validate_uniform_params([0, 0]) == (0.0, 0.0)


def test_validate_triangular_params():
    """Test triangular distribution parameter validator"""

    # Test invalid parameter counts
    with pytest.raises(ValueError, match="requires exactly 3 parameters"):
        dists.parse_and_validate_triangular_params([])

    with pytest.raises(ValueError, match="requires exactly 3 parameters"):
        dists.parse_and_validate_triangular_params(())

    with pytest.raises(ValueError, match="requires exactly 3 parameters"):
        dists.parse_and_validate_triangular_params([0])

    with pytest.raises(ValueError, match="requires exactly 3 parameters"):
        dists.parse_and_validate_triangular_params([0, 0])

    with pytest.raises(ValueError, match="requires exactly 3 parameters"):
        dists.parse_and_validate_triangular_params([0, 0, 0, 0])

    # Test non-numeric parameters
    with pytest.raises(ValueError, match="must be convertible to numbers"):
        dists.parse_and_validate_triangular_params(["foo", "bar", 0])

    # Test valid parameters
    low, mode, high = dists.parse_and_validate_triangular_params([0, 1, 2])
    assert (low, mode, high) == (0.0, 1.0, 2.0)

    # Test invalid ordering
    with pytest.raises(ValueError, match="must satisfy low <= mode <= high"):
        dists.parse_and_validate_triangular_params([0, -1, -4])

    with pytest.raises(ValueError, match="must satisfy low <= mode <= high"):
        dists.parse_and_validate_triangular_params([0, 1000, 999.99])

    # Test edge case (all equal values)
    low, mode, high = dists.parse_and_validate_triangular_params([0, 0, 0])
    assert (low, mode, high) == (0.0, 0.0, 0.0)


def test_check_dist_params_pert():
    """Test pert dist param checker"""
    assert not dists._check_dist_params_pert([])[0]
    assert not dists._check_dist_params_pert(())[0]

    assert not dists._check_dist_params_pert([0])[0]
    assert not dists._check_dist_params_pert([0, 0])[0]
    assert not dists._check_dist_params_pert([0, 0, 0, 0, 0])[0]

    assert not dists._check_dist_params_pert(["foo", "bar", 0])[0]

    assert dists._check_dist_params_pert([0, 1, 2])[0]
    assert not dists._check_dist_params_pert([0, -1, -4])[0]
    assert not dists._check_dist_params_pert([0, 1000, 999.99])[0]
    assert dists._check_dist_params_pert([0, 1000, 1000, 999.99])[0]

    assert dists._check_dist_params_pert([0, 0, 0])[0]  # edge case


class TestLoguniformDistribution:
    def test_that_empty_list_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Loguniform distribution requires exactly 2 parameters, got 0",
        ):
            dists.parse_and_validate_loguniform_params([])

    def test_that_empty_tuple_raises_parameter_count_error(self):
        with pytest.raises(
            ValueError,
            match="Loguniform distribution requires exactly 2 parameters, got 0",
        ):
            dists.parse_and_validate_loguniform_params(())

    def test_that_insufficient_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Loguniform distribution requires exactly 2 parameters, got 1",
        ):
            dists.parse_and_validate_loguniform_params([1])

    def test_that_excess_parameters_raises_count_error(self):
        with pytest.raises(
            ValueError,
            match="Loguniform distribution requires exactly 2 parameters, got 3",
        ):
            dists.parse_and_validate_loguniform_params([1, 2, 3])

    def test_that_non_numeric_parameters_raise_conversion_error(self):
        with pytest.raises(ValueError, match="must be convertible to numbers"):
            dists.parse_and_validate_loguniform_params(["foo", "bar"])

    def test_that_low_is_zero_raises_constraint_error(self):
        with pytest.raises(
            ValueError, match="For loguniform distribution, low must be > 0, got 0.0"
        ):
            dists.parse_and_validate_loguniform_params([0, 1])

    def test_that_low_is_negative_raises_constraint_error(self):
        with pytest.raises(
            ValueError, match="For loguniform distribution, low must be > 0, got -1.0"
        ):
            dists.parse_and_validate_loguniform_params([-1, 1])

    def test_that_invalid_ordering_raises_constraint_error(self):
        with pytest.raises(
            ValueError, match=r"Parameters must satisfy low <= high, got \[2.0, 1.0\]"
        ):
            dists.parse_and_validate_loguniform_params([2, 1])

    def test_that_valid_parameters_return_float_tuple(self):
        assert dists.parse_and_validate_loguniform_params([1, 2]) == (1.0, 2.0)

    def test_that_equal_bounds_are_accepted(self):
        assert dists.parse_and_validate_loguniform_params([1, 1]) == (1.0, 1.0)

    def test_that_nan_parameter_raises_error(self):
        with pytest.raises(ValueError, match="Parameters cannot be NaN"):
            dists.parse_and_validate_loguniform_params([1, np.nan])

    def test_that_string_numbers_are_converted_to_floats(self):
        assert dists.parse_and_validate_loguniform_params(["1", "2"]) == (1.0, 2.0)


def test_draw_values_normal():
    rng = np.random.default_rng()
    values = dists.draw_values_normal(["0", "1"], 10, rng)
    assert len(values) == 10
    assert all(isinstance(value, numbers.Number) for value in values)

    with pytest.raises(
        ValueError,
        match=(
            "Normal distribution must have 2 parameters or 4 for a truncated normal, "
            "but had 3 parameters."
        ),
    ):
        dists.draw_values_normal(["0", "1", "2"], 10, rng)

    with pytest.raises(
        ValueError,
        match=r"All parameters must be convertible to numbers. Got: \['0', 'b'\]",
    ):
        dists.draw_values_normal(["0", "b"], 10, rng)

    with pytest.raises(
        ValueError, match="Stddev for normal distribution must be >= 0. Got: -1.0"
    ):
        dists.draw_values_normal(["0", "-1"], 10, rng)

    with pytest.raises(
        ValueError,
        match=r"For truncated normal distribution, lower bound must be less than upper bound, but got \[2.0, -1.0\].",
    ):
        dists.draw_values_normal(["0", "1", "2", "-1"], 10, rng)

    values = dists.draw_values_normal(["0", "10", "-1", "2"], 50, rng)
    assert all(-1 <= value <= 2 for value in values)


def test_draw_values_uniform():
    rng = np.random.default_rng()

    values = dists.draw_values_uniform([10, 100], 0, rng)
    assert len(values) == 0

    values = dists.draw_values_uniform([10, 100], 20, rng)
    assert len(values) == 20
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(10 <= value <= 100 for value in values)

    with pytest.raises(
        ValueError,
        match="Uniform distribution requires exactly 2 parameters, got 3",
    ):
        values = dists.draw_values_uniform([10, 50, 100], 10, rng)

    with pytest.raises(ValueError, match="must satisfy low < high"):
        values = dists.draw_values_uniform([50, 10], 10, rng)

    with pytest.raises(ValueError, match="must be convertible to numbers"):
        values = dists.draw_values_uniform(["a", 10], 10, rng)

    with pytest.raises(ValueError, match="numreal must be a positive integer"):
        values = dists.draw_values_uniform([10, 50], -10, rng)


def test_draw_values_triangular():
    rng = np.random.default_rng()

    assert not dists.draw_values_triangular([10, 100, 1000], 0, rng).size
    with pytest.raises(ValueError):
        assert not dists.draw_values_triangular([10, 100, 1000], -1, rng)

    with pytest.raises(TypeError):
        assert not dists.draw_values_triangular([10, 100, 1000], "somestring", rng)

    values = dists.draw_values_triangular([10, 100, 1000], 15, rng)
    assert len(values) == 15
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(10 <= value <= 1000 for value in values)

    with pytest.raises(ValueError, match="minimum .* and maximum .* cannot be equal"):
        dists.draw_values_triangular([100, 100, 100], 10, rng)


@pytest.mark.parametrize("seed", range(100))
def test_draw_values_pert(seed):
    rng = np.random.default_rng(seed)

    assert not dists.draw_values_pert([10, 50, 100], 0, rng).size

    values = dists.draw_values_pert([10, 50, 100], 20, rng)
    assert len(values) == 20
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(10 <= value <= 100 for value in values)

    # For symmetric PERT, verify both mode and mean are at 5
    values = dists.draw_values_pert([0, 5, 10, 4], 5000, rng)
    hist, bins = np.histogram(values, bins=100)
    empirical_mode = bins[np.argmax(hist)]
    assert np.isclose(empirical_mode, 5, atol=0.55)
    assert np.isclose(values.mean(), 5, atol=0.55)


def test_draw_values_loguniform():
    rng = np.random.default_rng()

    assert not dists.draw_values_loguniform([10, 100], 0, rng).size

    values = dists.draw_values_loguniform([10, 100], 20, rng)
    assert len(values) == 20
    assert all(isinstance(value, numbers.Number) for value in values)
    assert all(10 <= value <= 100 for value in values)

    with pytest.raises(
        ValueError,
        match="Loguniform distribution requires exactly 2 parameters, got 3",
    ):
        dists.draw_values_loguniform([10, 50, 100], 10, rng)

    with pytest.raises(
        ValueError, match=r"Parameters must satisfy low <= high, got \[50.0, 10.0\]"
    ):
        dists.draw_values_loguniform([50, 10], 10, rng)

    with pytest.raises(ValueError, match="must be convertible to numbers"):
        dists.draw_values_loguniform(["a", 10], 10, rng)

    with pytest.raises(
        ValueError, match="For loguniform distribution, low must be > 0, got 0.0"
    ):
        dists.draw_values_loguniform([0, 10], 10, rng)


def test_sample_discrete():
    rng = np.random.default_rng()
    outcomes = ["foo", "bar.com"]
    # Test basic functionality
    values = dists.sample_discrete([",".join(outcomes)], 10, rng)[1]
    assert all(value in outcomes for value in values)

    # Test empty case
    assert not dists.sample_discrete([",".join(outcomes)], 0, rng)[1].size

    # Test negative numreals
    with pytest.raises(ValueError):
        dists.sample_discrete([",".join(outcomes)], -1, rng)[1]

    # Test weighted case where only bar.com should appear
    assert "foo" not in dists.sample_discrete([",".join(outcomes), "0,1"], 10, rng)[1]

    # Test weights that don't sum to 1
    weighted_values = dists.sample_discrete([",".join(outcomes), "2,6"], 100, rng)[1]
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
