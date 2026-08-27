from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from pydantic import TypeAdapter, ValidationError

from semeio.fmudesign.config_validation import SeedStrategy
from semeio.fmudesign.general_input import GeneralInput

PYDANTIC_PATH_ERROR = "Path does not point to a file|Input is not a valid path"


def base_general_input_dict():
    return {
        "designtype": "onebyone",
        "repeats": "10",
        "distribution_seed": None,
        "rms_seeds": None,
        "correlation_iterations": "1",
        "seed_strategy": SeedStrategy.JOINT,
        "background": None,
    }


def is_pydantic_numeric(x):
    for adapter in (TypeAdapter(int), TypeAdapter(float)):
        try:
            adapter.validate_python(x.strip())
            return True
        except ValidationError:
            pass
    return False


TEXT_STRIPPED = st.text().map(str.strip)
TEXT_STRIPPED_NOT_NUMERIC = TEXT_STRIPPED.filter(lambda x: not is_pydantic_numeric(x))
TEXT_STRIPPED_OR_NONE = st.one_of(TEXT_STRIPPED_NOT_NUMERIC, st.none())


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


def test_that_extra_key_raises_value_error():
    extra = {"extra_key": "foo"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GeneralInput.from_dict(base_general_input_dict() | extra)


def test_that_designtype_onebyone_is_accepted():
    result = GeneralInput.from_dict(base_general_input_dict())
    assert result.designtype == "onebyone"


@given(TEXT_STRIPPED)
def test_that_other_design_types_than_onebyone_raises_validation_error(text):
    assume(text != "onebyone")
    general_input_dict = base_general_input_dict() | {"designtype": text}
    with pytest.raises(ValueError, match="Input should be 'onebyone'"):
        GeneralInput.from_dict(general_input_dict)


@given(st.integers(min_value=1, max_value=10000))
def test_that_positive_int_repeats_is_accepted(positive_int):
    general_input_dict = base_general_input_dict() | {"repeats": str(positive_int)}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.repeats == positive_int


def test_that_zero_repeats_raises_validation_error():
    general_input_dict = base_general_input_dict() | {"repeats": "0"}
    with pytest.raises(ValidationError, match="repeats"):
        GeneralInput.from_dict(general_input_dict)


def test_that_negative_repeats_raises_validation_error():
    general_input_dict = base_general_input_dict() | {"repeats": "-1"}
    with pytest.raises(ValidationError):
        GeneralInput.from_dict(general_input_dict)


@given(TEXT_STRIPPED_NOT_NUMERIC)
def test_that_non_integer_repeats_raises_validation_error(non_int):
    general_input_dict = base_general_input_dict() | {"repeats": str(non_int)}
    with pytest.raises(
        ValidationError,
        match=r"Input should be greater than 0|Input should be a valid integer",
    ):
        GeneralInput.from_dict(general_input_dict)


@given(st.integers(min_value=0))
def test_that_non_negative_distribution_seed_is_accepted(value):
    general_input_dict = base_general_input_dict() | {"distribution_seed": str(value)}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.distribution_seed == value


def test_that_none_distribution_seed_is_accepted():
    general_input_dict = base_general_input_dict() | {"distribution_seed": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.distribution_seed is None


@given(st.integers(max_value=-1))
def test_that_negative_distribution_seed_raises_validation_error(value):
    general_input_dict = base_general_input_dict() | {"distribution_seed": str(value)}
    with pytest.raises(ValidationError):
        GeneralInput.from_dict(general_input_dict)


@given(TEXT_STRIPPED_NOT_NUMERIC)
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


def test_that_rms_seeds_existing_file_is_accepted(use_tmpdir):
    seeds_file_name = "seeds.csv"
    Path(seeds_file_name).touch()
    general_input_dict = base_general_input_dict() | {
        "rms_seeds": seeds_file_name,
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert result.rms_seeds == Path(seeds_file_name).resolve()


def test_that_rms_seeds_non_existing_file_raises(use_tmpdir):
    seeds_file_name = "seeds.csv"
    general_input_dict = base_general_input_dict() | {
        "rms_seeds": seeds_file_name,
    }
    with pytest.raises(ValidationError, match=r"Path does not point to a file"):
        GeneralInput.from_dict(general_input_dict)


def test_that_rms_seeds_from_extern_csv_file_in_subdirectory_is_accepted(use_tmpdir):
    subdir = Path("subdir")
    subdir.mkdir()
    seeds_file_name = "seeds.csv"
    (subdir / seeds_file_name).touch()
    general_input_dict = base_general_input_dict() | {
        "rms_seeds": str(seeds_file_name),
    }
    result = GeneralInput.from_dict(
        general_input_dict, input_filename=str(subdir / "input.xlsx")
    )
    assert result.rms_seeds == (subdir / seeds_file_name).resolve()


@given(TEXT_STRIPPED)
def test_that_invalid_rms_seeds_types_raises_validation_error(invalid_rms_seeds):
    assume(invalid_rms_seeds != "default")
    general_input_dict = base_general_input_dict() | {"rms_seeds": invalid_rms_seeds}
    with pytest.raises(
        ValueError, match=r"Path does not point to a file|Input is not a valid path"
    ):
        GeneralInput.from_dict(general_input_dict)


def test_that_rms_seeds_from_external_txt_file_is_resolved(use_tmpdir):
    seeds_file_name = "seeds.txt"
    Path(seeds_file_name).write_text("1\n2\n3", encoding="utf-8")
    gi = GeneralInput.from_dict(
        base_general_input_dict() | {"rms_seeds": seeds_file_name}
    )
    assert gi.rms_seeds == Path(seeds_file_name).resolve()


def test_that_none_correlation_iterations_defaults_to_zero():
    general_input_dict = base_general_input_dict() | {"correlation_iterations": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.correlation_iterations == 0


def test_that_missing_correlation_iterations_defaults_to_zero():
    general_input_dict = base_general_input_dict()
    general_input_dict.pop("correlation_iterations")
    result = GeneralInput.from_dict(general_input_dict)
    assert result.correlation_iterations == 0


@given(st.integers(min_value=0))
def test_that_non_negative_correlation_iterations_is_accepted(value):
    general_input_dict = base_general_input_dict() | {
        "correlation_iterations": str(value),
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert result.correlation_iterations == value


@given(TEXT_STRIPPED_NOT_NUMERIC)
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
        "correlation_iterations": "-1",
    }
    with pytest.raises(ValidationError):
        GeneralInput.from_dict(general_input_dict)


def test_that_none_seed_strategy_defaults_to_joint():
    general_input_dict = base_general_input_dict() | {"seed_strategy": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.seed_strategy == SeedStrategy.JOINT


def test_that_missing_seed_strategy_defaults_to_joint():
    general_input_dict = base_general_input_dict()
    general_input_dict.pop("seed_strategy")
    result = GeneralInput.from_dict(general_input_dict)
    assert result.seed_strategy == SeedStrategy.JOINT


def test_that_seed_strategy_defaults_to_string_type():
    general_input_dict = base_general_input_dict()
    general_input_dict.pop("seed_strategy")
    result = GeneralInput.from_dict(general_input_dict)
    assert type(result.seed_strategy) is str


@pytest.mark.parametrize("strategy", list(SeedStrategy))
def test_that_valid_seed_strategies_are_accepted(strategy):
    general_input_dict = base_general_input_dict() | {"seed_strategy": strategy.value}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.seed_strategy == strategy


TEXT_OR_NONE_NOT_IN_SEED_STRATEGY = TEXT_STRIPPED_OR_NONE.filter(
    lambda x: isinstance(x, str) and x not in SeedStrategy
)


@given(TEXT_OR_NONE_NOT_IN_SEED_STRATEGY)
def test_that_invalid_seed_strategy_raises_validation_error(invalid_seed_strategy):
    general_input_dict = base_general_input_dict() | {
        "seed_strategy": invalid_seed_strategy
    }
    seed_strategy_error = "Input should be 'joint' or 'independent'"
    with pytest.raises(ValueError, match=seed_strategy_error):
        GeneralInput.from_dict(general_input_dict)


def test_that_seed_strategy_is_serialized_as_str():
    gi = GeneralInput.from_dict(base_general_input_dict())
    assert type(gi.seed_strategy) is str


def test_that_background_none_is_accepted():
    general_input_dict = base_general_input_dict() | {"background": None}
    result = GeneralInput.from_dict(general_input_dict)
    assert result.background is None


def test_that_existing_background_csv_path_is_accepted(use_tmpdir):
    bg_file = "background.csv"
    Path(bg_file).touch()
    general_input_dict = base_general_input_dict() | {
        "background": bg_file,
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert result.background == Path(bg_file).resolve()


def test_that_non_existing_background_csv_path_is_accepted_as_str():
    """Background is either file (Path) or excel sheet name (str).
    The validation should allow strings as it may be a valid background sheet.
    Whether this is the case is responsibility outside of the scope of GeneralInput."""
    bg_file = "background"
    general_input_dict = base_general_input_dict() | {
        "background": bg_file,
    }
    result = GeneralInput.from_dict(general_input_dict)
    assert isinstance(result.background, str)
    assert result.background == bg_file


def test_that_background_file_with_rel_path_to_input_file_is_resolved_to_path_from_cwd(
    use_tmpdir,
):
    """Tests that background file with relative path to input_filename is resolved
    to path."""
    subdir = Path("subdir")
    subdir.mkdir()
    input_file = subdir / "input.csv"
    bg_file = "background.csv"
    (subdir / bg_file).touch()
    result = GeneralInput.from_dict(
        base_general_input_dict()
        | {
            "background": bg_file,
        },
        input_filename=str(input_file),
    )
    assert result.background == Path(subdir / bg_file).resolve()


def test_that_background_path_is_serialized_as_string():
    bg_file = "background.csv"
    Path(bg_file).touch()
    general_input_dict = base_general_input_dict() | {
        "background": bg_file,
    }
    gi = GeneralInput.from_dict(general_input_dict)
    assert isinstance(gi.background, Path)
    assert isinstance(gi.model_dump()["background"], str)


def test_that_serialize_paths_doesnt_fail_given_none():
    gi = GeneralInput.from_dict(base_general_input_dict())
    assert gi.model_dump()["background"] is None
