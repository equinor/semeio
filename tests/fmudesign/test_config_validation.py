"""Tests for validation of 'seed_strategy' in the general_input sheet."""

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given

from semeio.fmudesign.config_validation import (
    ConfigValidationError,
    SeedStrategy,
    validate_configuration,
)


def _minimal_config(**extra):
    return {
        "designtype": "onebyone",
        "repeats": 10,
        "distribution_seed": 42,
        "seeds": "default",
        **extra,
    }


def test_seed_strategy_defaults_to_joint():
    cfg = validate_configuration(_minimal_config())
    assert cfg["seed_strategy"] is SeedStrategy.JOINT


def test_seed_strategy_is_normalized_to_enum_member():
    cfg = validate_configuration(_minimal_config(seed_strategy="independent"))
    assert cfg["seed_strategy"] is SeedStrategy.INDEPENDENT
    assert cfg["seed_strategy"] == "independent"


def test_seed_strategy_invalid_raises():
    with pytest.raises(ValueError, match="seed_strategy"):
        validate_configuration(_minimal_config(seed_strategy="bogus"))


@pytest.mark.parametrize("value", [["independent"], {"joint": 1}, 5, 1.5])
def test_seed_strategy_non_string_raises_value_error(value):
    """Unsupported types must be rejected as validation errors, not TypeErrors."""
    with pytest.raises(ValueError, match="seed_strategy"):
        validate_configuration(_minimal_config(seed_strategy=value))


@pytest.mark.parametrize("value", ["Independent", "INDEPENDENT", " independent "])
def test_seed_strategy_is_case_and_whitespace_insensitive(value):
    """Excel auto-capitalizes cell text, so 'Independent' must be accepted."""
    cfg = validate_configuration(_minimal_config(seed_strategy=value))
    assert cfg["seed_strategy"] is SeedStrategy.INDEPENDENT


@pytest.mark.parametrize("value", [None, "None"])
def test_seed_strategy_none_falls_back_to_joint(value):
    cfg = validate_configuration(_minimal_config(seed_strategy=value))
    assert cfg["seed_strategy"] is SeedStrategy.JOINT


def _setup_and_validate_config(design_type="onebyone", repeat=5, extra_keys=None):
    return validate_configuration(
        {
            "designtype": design_type,
            "repeats": repeat,
            "distribution_seed": None,
            "seeds": None,
        }
        | (extra_keys or {})
    )


def test_that_string_repeat_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(repeat="foo")


def test_that_none_repeat_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(repeat=None)


def test_that_list_repeat_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(repeat=[1, 2])


def test_that_int_repeat_does_not_raise_config_validation_error():
    config = _setup_and_validate_config(repeat=5)
    assert isinstance(config["repeats"], int)


def test_that_int_design_string_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(design_type=123)


def test_that_float_design_string_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(design_type=7.5)


def test_that_none_design_string_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(design_type=None)


@given(st.text())
def test_that_any_design_string_not_one_by_one_raises_config_validation_error(
    designtype,
):
    assume(designtype != "onebyone")
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(design_type=designtype)


def test_that_missing_correlation_iteration_defaults_to_zerio():
    validated = _setup_and_validate_config()
    assert validated["correlation_iterations"] == 0


def test_that_str_correlation_iteration_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(extra_keys={"correlation_iterations": "foo"})


def test_that_negative_float_correlation_iteration_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(extra_keys={"correlation_iterations": -5.5})


def test_that_negative_int_correlation_iteration_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(extra_keys={"correlation_iterations": -5})


def test_that_positive_float_correlation_iteration_does_not_raise_validation_error():
    _setup_and_validate_config(extra_keys={"correlation_iterations": 5.5})


def test_that_positive_int_correlation_iteration_does_not_raise_validation_error():
    _setup_and_validate_config(extra_keys={"correlation_iterations": 5})


def test_that_negative_distribution_seed_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(extra_keys={"distribution_seed": -1234})


def test_that_positive_distribution_seed_does_not_raise_config_validation_error():
    _setup_and_validate_config(extra_keys={"distribution_seed": 1234})


def test_that_none_distribution_seed_does_not_raise_config_validation_error():
    _setup_and_validate_config(extra_keys={"distribution_seed": None})


def test_that_string_distribution_seed_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _setup_and_validate_config(extra_keys={"distribution_seed": "foo"})
