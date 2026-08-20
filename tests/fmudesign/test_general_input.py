import re
from pathlib import Path
from types import NoneType

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from pydantic import ValidationError

from semeio.fmudesign._excel_to_dict import GeneralInput
from semeio.fmudesign.config_validation import SeedStrategy
from semeio.fmudesign.general_input import parse_value


def base_general_input_dict():
    return {
        "input_filename": "foo.xlsx",
        "designtype": "onebyone",
        "repeats": 10,
        "distribution_seed": None,
        "rms_seeds": None,
        "correlation_iterations": 1,
        "seed_strategy": SeedStrategy.JOINT,
        "background": None,
    }


ANY_TYPE = st.one_of(
    st.integers(),
    st.floats(),
    st.text(),
    st.booleans(),
    st.none(),
    st.lists(st.integers(), min_size=1),
    st.dictionaries(st.text(), st.integers() | st.text(), min_size=1),
)

NON_NUMERIC = ANY_TYPE.filter(
    lambda x: not (isinstance(x, int | float) or (isinstance(x, str) and x.isnumeric()))
)


BOOLEAN_ERROR = "cannot have boolean value"


@pytest.mark.parametrize(
    "nan_value",
    [
        float("nan"),
        np.nan,
        np.float64("nan"),
        pd.NaT,
        None,
    ],
)
def test_that_parse_value_converts_nan_formats_to_none(nan_value):
    assert parse_value(nan_value) is None


@pytest.mark.parametrize(
    "required_key",
    (key for key, info in GeneralInput.model_fields.items() if info.is_required()),
)
def test_that_missing_required_keys_raises_validation_error(required_key):
    general_input_dict = base_general_input_dict()
    general_input_dict.pop(required_key)
    with pytest.raises(ValidationError):
        GeneralInput.from_dict(general_input_dict)


@pytest.mark.parametrize(
    "optional_key",
    (key for key, info in GeneralInput.model_fields.items() if not info.is_required()),
)
def test_that_missing_optional_keys_does_not_raise_validation_error(optional_key):
    general_input_dict = base_general_input_dict()
    general_input_dict.pop(optional_key)
    GeneralInput.from_dict(general_input_dict)


def test_that_unknown_key_raises_value_error():
    general_input_dict = base_general_input_dict() | {"unknown_key": 42}
    with pytest.raises(
        ValueError, match=re.escape("Invalid key 'unknown_key' in general_input.")
    ):
        GeneralInput.from_dict(general_input_dict)


def test_that_input_file_key_is_silently_skipped():
    general_input_dict = base_general_input_dict()
    result = GeneralInput.from_dict(general_input_dict)
    assert not hasattr(result, "input_filename")


def test_that_whitespace_around_keys_is_stripped():
    general_input_dict = base_general_input_dict()
    general_input_dict[" repeats "] = general_input_dict.pop("repeats")
    result = GeneralInput.from_dict(general_input_dict)
    assert result.repeats == 10


def test_that_designtype_onebyone_is_accepted():
    result = GeneralInput.from_dict(base_general_input_dict())
    assert result.designtype == "onebyone"


@given(ANY_TYPE)
def test_that_other_design_types_than_onebyone_raises_validation_error(text):
    assume(text != "onebyone")
    general_input_dict = base_general_input_dict() | {"designtype": text}
    with pytest.raises(ValueError, match=f"Input should be 'onebyone'|{BOOLEAN_ERROR}"):
        GeneralInput.from_dict(general_input_dict)


@given(st.integers(min_value=1, max_value=10000))
def test_that_positive_int_repeats_is_accepted(positive_int):
    general_input_dict = base_general_input_dict() | {"repeats": positive_int}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.repeats == positive_int


def test_that_zero_repeats_raises_validation_error():
    general_input_dict = base_general_input_dict() | {"repeats": 0}
    with pytest.raises(ValidationError, match="repeats"):
        GeneralInput.from_dict(general_input_dict)


def test_that_negative_repeats_raises_validation_error():
    general_input_dict = base_general_input_dict() | {"repeats": -1}
    with pytest.raises(
        ValidationError,
        match="'repeats' in generalinput must be an int greater than zero",
    ):
        GeneralInput.from_dict(general_input_dict)


@given(NON_NUMERIC)
def test_that_non_integer_repeats_raises_validation_error(non_int):
    general_input_dict = base_general_input_dict() | {"repeats": non_int}
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        GeneralInput.from_dict(general_input_dict)


@given(st.integers(min_value=0))
def test_that_non_negative_distribution_seed_is_accepted(value):
    general_input_dict = base_general_input_dict() | {"distribution_seed": value}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.distribution_seed == value


def test_that_none_distribution_seed_is_accepted():
    general_input_dict = base_general_input_dict() | {"distribution_seed": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.distribution_seed is None


@given(st.integers(max_value=-1))
def test_that_negative_distribution_seed_raises_validation_error(value):
    general_input_dict = base_general_input_dict() | {"distribution_seed": value}
    with pytest.raises(
        ValidationError,
        match="'distribution_seed' in generalinput must be a positive int",
    ):
        GeneralInput.from_dict(general_input_dict)


@given(NON_NUMERIC.filter(lambda x: not isinstance(x, NoneType)))
def test_that_invalid_distribution_seed_types_raises_validation_error(
    invalid_distribution_seed,
):
    general_input_dict = base_general_input_dict() | {
        "distribution_seed": invalid_distribution_seed
    }
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        GeneralInput.from_dict(general_input_dict)


def test_that_rms_seeds_none_is_accepted():
    general_input_dict = base_general_input_dict() | {"rms_seeds": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.rms_seeds is None


def test_that_rms_seeds_default_is_accepted():
    general_input_dict = base_general_input_dict() | {"rms_seeds": "default"}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.rms_seeds == "default"


def test_that_rms_seeds_from_extern_csv_file_is_accepted(use_tmpdir):
    seeds_file_name = "seeds.csv"
    seeds_file = Path(seeds_file_name)
    seeds_file.write_text("100\n200\n300\n", encoding="utf-8")
    workbook = "input.xlsx"
    general_input_dict = base_general_input_dict() | {
        "input_filename": workbook,
        "rms_seeds": seeds_file_name,
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert result.rms_seeds == [100, 200, 300]


def test_that_rms_seeds_from_extern_csv_file_in_subdirectory_is_accepted(use_tmpdir):
    subdir = Path("subdir")
    subdir.mkdir()
    seeds_file = subdir / "seeds.csv"
    seeds_file.write_text("100\n200\n300\n", encoding="utf-8")
    general_input_dict = base_general_input_dict() | {
        "input_filename": "input.xlsx",
        "rms_seeds": str(seeds_file),
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert result.rms_seeds == [100, 200, 300]


def test_that_rms_seeds_nonexistent_file_string_raises():
    general_input_dict = base_general_input_dict() | {
        "rms_seeds": "no_such_file.csv",
    }
    with pytest.raises(ValueError, match="Failed to resolve path"):
        GeneralInput.from_dict(general_input_dict)


rms_seeds_type = GeneralInput.model_fields["rms_seeds"].annotation


@given(ANY_TYPE.filter(lambda x: not isinstance(x, list | NoneType)))
def test_that_invalid_rms_seeds_types_raises_validation_error(invalid_rms_seeds):
    assume(invalid_rms_seeds != "default")
    general_input_dict = base_general_input_dict() | {"rms_seeds": invalid_rms_seeds}
    validate_path_error = (
        "External file with seed values should be on Excel "
        "or csv format and end with .xlsx .csv or .txt"
    )
    value_error = "must be 'default', 'None' or relative path"
    with pytest.raises(
        ValueError,
        match=f"{validate_path_error}|{value_error}|{BOOLEAN_ERROR}",
    ):
        GeneralInput.from_dict(general_input_dict)


def test_that_correlation_iterations_defaults_to_zero():
    general_input_dict = base_general_input_dict()
    general_input_dict.pop("correlation_iterations")
    result = GeneralInput.from_dict(general_input_dict)
    assert result.correlation_iterations == 0


@given(st.integers(min_value=0))
def test_that_non_negative_correlation_iterations_is_accepted(value):
    general_input_dict = base_general_input_dict() | {
        "correlation_iterations": value,
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert result.correlation_iterations == value


@given(NON_NUMERIC)
def test_that_invalid_correlation_iterations_raises_validation_error(
    invalid_correlation_iterations,
):
    general_input_dict = base_general_input_dict() | {
        "correlation_iterations": invalid_correlation_iterations,
    }
    with pytest.raises(ValidationError):
        GeneralInput.from_dict(general_input_dict)


def test_that_negative_correlation_iterations_raises_validation_error():
    general_input_dict = base_general_input_dict() | {
        "correlation_iterations": -1,
    }
    with pytest.raises(
        ValidationError,
        match="'correlation_iterations' in generalinput must be a positive int or zero",
    ):
        GeneralInput.from_dict(general_input_dict)


def test_that_seed_strategy_defaults_to_joint():
    general_input_dict = base_general_input_dict()
    general_input_dict.pop("seed_strategy")
    result = GeneralInput.from_dict(general_input_dict)
    assert result.seed_strategy == SeedStrategy.JOINT


@pytest.mark.parametrize("strategy", list(SeedStrategy))
def test_that_valid_seed_strategies_are_accepted(strategy):
    general_input_dict = base_general_input_dict() | {"seed_strategy": strategy.value}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.seed_strategy == strategy


@given(
    ANY_TYPE.filter(lambda x: all(x != seed_strategy for seed_strategy in SeedStrategy))
)
def test_that_invalid_seed_strategy_raises_validation_error(invalid_seed_strategy):
    general_input_dict = base_general_input_dict() | {
        "seed_strategy": invalid_seed_strategy
    }
    bool_error = "key 'seed_strategy' cannot have boolean value"
    seed_strategy_error = "should be 'joint' or 'independent'"
    with pytest.raises(ValueError, match=f"{bool_error}|{seed_strategy_error}"):
        GeneralInput.from_dict(general_input_dict)


def test_that_background_none_is_accepted():
    general_input_dict = base_general_input_dict() | {"background": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.background is None


@pytest.mark.parametrize("value", ["None", "none", "NONE", ""])
def test_that_background_none_like_strings_are_treated_as_none(value):
    general_input_dict = base_general_input_dict() | {"background": value}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.background is None


def test_that_background_csv_path_creates_extern_dict(use_tmpdir):
    bg_file = "background.csv"
    Path(bg_file).write_text("col1\n1\n2\n", encoding="utf-8")
    workbook = "input.xlsx"
    general_input_dict = base_general_input_dict() | {
        "input_filename": str(workbook),
        "background": "background.csv",
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert isinstance(result.background, dict)
    assert "extern" in result.background


def test_that_background_xlsx_path_creates_extern_dict(tmp_path):
    bg_file = tmp_path / "background.xlsx"
    bg_file.write_text("")  # just needs to exist for resolve_path
    workbook = tmp_path / "input.xlsx"
    general_input_dict = base_general_input_dict() | {
        "input_filename": str(workbook),
        "background": "background.xlsx",
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert isinstance(result.background, dict)
    assert "extern" in result.background


def test_that_background_in_subfolder_creates_extern_dict(use_tmpdir):
    subdir = Path("subdir")
    subdir.mkdir()
    bg_file = subdir / "background.csv"
    bg_file.write_text("col1\n1\n2\n", encoding="utf-8")
    workbook = "input.xlsx"
    general_input_dict = base_general_input_dict() | {
        "input_filename": str(workbook),
        "background": str(bg_file),
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert isinstance(result.background, dict)
    assert "extern" in result.background
