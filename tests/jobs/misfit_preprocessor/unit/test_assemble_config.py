import pytest

from semeio.workflows.misfit_preprocessor.exceptions import ValidationError
from semeio.workflows.misfit_preprocessor import assemble_config


@pytest.mark.parametrize(
    "input_observations, expected_result",
    [
        (["o*"], ["observations", "of"]),
        (["of"], ["of"]),
        (["a", "list"], ["a", "list"]),
    ],
)
def test_assemble_config(input_observations, expected_result):
    config = assemble_config(
        {"observations": input_observations},
        ["a", "list", "of", "existing", "observations"],
    )

    assert expected_result == sorted(config.observations)


def test_assemble_config_default_observations():
    config = assemble_config(
        {},
        ["a", "list", "of", "existing", "observations"],
    )

    assert sorted(["a", "list", "of", "existing", "observations"]) == sorted(
        config.observations
    )


def test_assemble_config_not_existing_obs():
    with pytest.raises(ValidationError) as ve:
        assemble_config(
            {"observations": ["not_an_observation"]},
            ["a", "list", "of", "existing", "observations"],
        )

    expected_err_msg = (
        "Invalid configuration of misfit preprocessor\n"
        "  - Found no match for observation not_an_observation (observations)\n"
    )
    assert str(ve.value) == expected_err_msg
