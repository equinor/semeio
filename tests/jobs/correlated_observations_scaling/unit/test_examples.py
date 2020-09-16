import configsuite
import pytest
import rstcheck

from semeio.workflows.correlated_observations_scaling import cos, job_config


@pytest.mark.parametrize(
    "input_config",
    [
        cos._calc_keys_example,
        cos._calc_keys_w_index,
        cos._wildcard_example,
        cos._all_keys_w_index,
        *cos._groups_example,
    ],
)
def test_valid_examples(input_config):
    default_values = {
        "CALCULATE_KEYS": {"std_cutoff": 1.0e-6, "alpha": 3},
        "UPDATE_KEYS": {},
    }
    schema = job_config._CORRELATED_OBSERVATIONS_SCHEMA
    config = configsuite.ConfigSuite(
        input_config,
        schema,
        layers=(default_values,),
        deduce_required=True,
    )
    assert config.valid


@pytest.mark.parametrize("input_rst", [cos._DESCRIPTION, cos._EXAMPLES])
def test_valid_rst(input_rst):
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck.check(input_rst))
