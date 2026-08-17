"""Tests for validation of 'seed_strategy' in the general_input sheet."""

import pytest

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


def _validate_config_with_repeat(repeat):
    return validate_configuration(
        {
            "designtype": "onebyone",
            "repeats": repeat,
            "distribution_seed": None,
            "seeds": None,
        }
    )


def test_that_string_repeat_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _validate_config_with_repeat("foo")


def test_that_none_repeat_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _validate_config_with_repeat(None)


def test_that_list_repeat_raises_config_validation_error():
    with pytest.raises(ConfigValidationError):
        _validate_config_with_repeat([1, 2])


def test_that_int_repeat_does_not_raise_config_validation_error():
    config = _validate_config_with_repeat(5)
    assert isinstance(config["repeats"], int)
