import pytest
from hypothesis import strategies as st, given
import pydantic
from semeio.workflows.localisation.localisation_config import (
    GaussianConfig,
    CorrelationConfig,
)


@given(
    st.one_of(
        st.integers(min_value=0, max_value=360),
        st.floats(
            min_value=0, max_value=360, width=64, exclude_min=False, exclude_max=False
        ),
    )
)
def test_gaussian_config_valid_angle(angle):
    config = GaussianConfig(
        method="gaussian_decay", main_range=0.1, perp_range=0.1, angle=angle
    )
    assert config.angle == angle


@pytest.mark.parametrize(
    "angle, expected_error",
    [
        (-0.0001, "ensure this value is greater than or equal to 0.0"),
        (360.0001, "ensure this value is less than or equal to 360"),
    ],
)
def test_gaussian_config_invalid_angle(angle, expected_error):
    with pytest.raises(pydantic.ValidationError, match=expected_error):
        GaussianConfig(
            method="gaussian_decay", main_range=0.1, perp_range=0.1, angle=angle
        )


@pytest.mark.parametrize("decay_method", ["gaussian_decay", "exponential_decay"])
def test_field_config_init(decay_method):
    field_config = {
        "method": decay_method,
        "main_range": 0.1,
        "perp_range": 0.1,
        "angle": 10,
    }
    obs_config = {
        "add": "*",
    }
    param_group = {"add": "*"}
    config = CorrelationConfig(
        name="a_name",
        obs_group=obs_config,
        param_group=param_group,
        field_scale=field_config,
    )
    assert config.field_scale.method == decay_method


def test_invalid_field_config_init():
    field_config = {
        "method": "not_implemented_method",
        "main_range": 0.1,
        "perp_range": 0.1,
        "angle": 10,
    }
    obs_config = {
        "add": "*",
    }
    param_group = {"add": "*"}
    with pytest.raises(
        pydantic.ValidationError, match="Unknown method: not_implemented_method"
    ):
        CorrelationConfig(
            name="a_name",
            obs_group=obs_config,
            param_group=param_group,
            field_scale=field_config,
        )
