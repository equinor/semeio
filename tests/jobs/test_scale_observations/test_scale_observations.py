from datetime import datetime

import pytest

from semeio.workflows.correlated_observations_scaling.update_scaling import (
    scale_observations,
)


class Config:  # pylint: disable=too-few-public-methods
    def __init__(self, key, index):
        self.key = key
        self.index = index


@pytest.fixture(name="snake_oil_obs")
def fixture_snake_oil_obs(snake_oil_facade):
    return snake_oil_facade.get_observations()


@pytest.mark.parametrize(
    "index_list",
    [None, [datetime(2010, 1, 10), datetime(2010, 1, 30), datetime(2010, 2, 9, 0, 0)]],
)
def test_scale_history_summary_obs(snake_oil_obs, index_list):
    scale_observations(snake_oil_obs, 1.2345, [Config("FOPR", index_list)])

    obs_vector = snake_oil_obs["FOPR"]
    for date, node in obs_vector.observations.items():
        if not index_list or date in index_list:
            assert node.std_scaling == 1.2345, f"index: {date}"
        else:
            assert node.std_scaling == 1.0, f"index: {date}"


@pytest.mark.parametrize("index_list", [None, [datetime(2010, 12, 26)]])
def test_scale_summary_obs(snake_oil_obs, index_list):
    scale_observations(snake_oil_obs, 1.2345, [Config("WOPR_OP1_36", index_list)])

    obs_vector = snake_oil_obs["WOPR_OP1_36"]
    node = obs_vector.observations[datetime(2010, 12, 26)]
    assert node.std_scaling == 1.2345, f"index: {datetime(2010, 12, 26)}"


@pytest.mark.parametrize("index_list", [None, [400, 800]])
def test_scale_gen_obs(snake_oil_obs, index_list):
    scale_observations(snake_oil_obs, 1.2345, [Config("WPR_DIFF_1", index_list)])

    obs_vector = snake_oil_obs["WPR_DIFF_1"]
    for index, node in enumerate(obs_vector):
        if not index_list or node.indices[index] in index_list:
            assert node.std_scaling[index] == 1.2345, f"index: {index}"
        else:
            assert node.std_scaling[index] == 1.0, f"index: {index}"
