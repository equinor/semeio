import pytest
from hypothesis import strategies as st, given
import pydantic
import pathlib
from semeio.workflows.localisation.localisation_config import (
    GaussianConfig,
    CorrelationConfig,
    ExponentialConfig,
)
from unittest.mock import MagicMock


@given(
    azimuth=st.one_of(
        st.integers(min_value=0, max_value=360),
        st.floats(
            min_value=0, max_value=360, width=64, exclude_min=False, exclude_max=False
        ),
    )
)
@pytest.mark.parametrize(
    "config_class, method",
    [(GaussianConfig, "gaussian_decay"), (ExponentialConfig, "exponential_decay")],
)
def test_gaussian_config_valid_angle(azimuth, config_class, method):
    config = config_class(
        method=method, main_range=0.1, perp_range=0.1, azimuth=azimuth
    )
    assert config.azimuth == azimuth


@pytest.mark.parametrize(
    "config_class, method",
    [(GaussianConfig, "gaussian_decay"), (ExponentialConfig, "exponential_decay")],
)
@pytest.mark.parametrize(
    "azimuth, expected_error",
    [
        (-0.0001, "ensure this value is greater than or equal to 0.0"),
        (360.0001, "ensure this value is less than or equal to 360"),
    ],
)
def test_invalid_angle(config_class, method, azimuth, expected_error):
    with pytest.raises(pydantic.ValidationError, match=expected_error):
        config_class(method=method, main_range=0.1, perp_range=0.1, azimuth=azimuth)


@pytest.mark.parametrize("decay_method", ["gaussian_decay", "exponential_decay"])
def test_field_config_init(decay_method):
    field_config = {
        "method": decay_method,
        "main_range": 0.1,
        "perp_range": 0.1,
        "azimuth": 10,
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
        ref_point=[10, 10],
        obs_context=["a", "b", "c"],
        params_context=["a", "b", "c"],
    )
    assert config.field_scale.method == decay_method


@pytest.mark.parametrize(
    "field_config, surface_config",
    [
        pytest.param(
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
            },
            None,
            id="Field scale, not surface scale",
        ),
        pytest.param(
            None,
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
                "surface_file": "surface_file.txt",
            },
            id="Surface scale, not field scale",
        ),
        pytest.param(
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
            },
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
                "surface_file": "surface_file.txt",
            },
            id="Field scale and surface scale",
        ),
    ],
)
def test_correlation_config_ref_point(field_config, surface_config, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "exists", MagicMock())

    config_dict = {
        "name": "random",
        "obs_group": {
            "add": "*",
        },
        "param_group": {"add": "*"},
        "ref_point": [1, 1],
        "obs_context": ["a", "b", "c"],
        "params_context": ["a", "b", "c"],
    }
    if field_config:
        config_dict["field_scale"] = field_config
    if surface_config:
        config_dict["surface_scale"] = surface_config
    config = CorrelationConfig(**config_dict)
    assert config.ref_point == [1, 1]


@pytest.mark.parametrize(
    "field_config, surface_config",
    [
        pytest.param(
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
            },
            None,
            id="Field scale, not surface scale",
        ),
        pytest.param(
            None,
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
                "surface_file": "surface_file.txt",
            },
            id="Surface scale, not field scale",
        ),
        pytest.param(
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
            },
            {
                "method": "gaussian_decay",
                "main_range": 0.1,
                "perp_range": 0.1,
                "azimuth": 10,
                "surface_file": "surface_file.txt",
            },
            id="Field scale and surface scale",
        ),
    ],
)
def test_correlation_config_no_ref_point(field_config, surface_config, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "exists", MagicMock())

    config_dict = {
        "name": "random",
        "obs_group": {
            "add": "*",
        },
        "param_group": {"add": "*"},
        "obs_context": ["a", "b", "c"],
        "params_context": ["a", "b", "c"],
    }
    if field_config:
        config_dict["field_scale"] = field_config
    if surface_config:
        config_dict["surface_scale"] = surface_config
    with pytest.raises(ValueError, match="the reference point must be specified"):
        CorrelationConfig(**config_dict)


def test_invalid_field_config_init():
    field_config = {
        "method": "not_implemented_method",
        "main_range": 0.1,
        "perp_range": 0.1,
        "azimuth": 10,
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
            obs_context=["a", "b", "c"],
            params_context=["a", "b", "c"],
        )
