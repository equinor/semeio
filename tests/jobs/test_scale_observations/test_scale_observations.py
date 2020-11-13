from res.enkf import EnKFMain
from semeio.workflows.correlated_observations_scaling.update_scaling import (
    scale_observations,
)
import pytest


class Config:  # pylint: disable=too-few-public-methods
    def __init__(self, key, index):
        self.key = key
        self.index = index


@pytest.fixture()
def snake_oil_obs(setup_ert):
    res_config = setup_ert
    ert = EnKFMain(res_config)
    yield ert.getObservations()


@pytest.mark.parametrize("index_list", [None, [0, 1, 2, 3]])
def test_scale_history_summary_obs(snake_oil_obs, index_list):

    scale_observations(snake_oil_obs, 1.2345, [Config("FOPR", index_list)])

    obs_vector = snake_oil_obs["FOPR"]
    for index, node in enumerate(obs_vector):
        if not index_list or index in index_list:
            assert node.getStdScaling(index) == 1.2345, f"index: {index}"
        else:
            assert node.getStdScaling(index) == 1.0, f"index: {index}"


@pytest.mark.parametrize("index_list", [None, [35]])
def test_scale_summary_obs(snake_oil_obs, index_list):

    scale_observations(snake_oil_obs, 1.2345, [Config("WOPR_OP1_36", index_list)])

    obs_vector = snake_oil_obs["WOPR_OP1_36"]
    node = obs_vector.getNode(36)
    assert node.getStdScaling(36) == 1.2345, f"index: {36}"


@pytest.mark.parametrize("index_list", [None, [400, 800]])
def test_scale_gen_obs(snake_oil_obs, index_list):

    scale_observations(snake_oil_obs, 1.2345, [Config("WPR_DIFF_1", index_list)])

    obs_vector = snake_oil_obs["WPR_DIFF_1"]
    for index, node in enumerate(obs_vector):
        if not index_list or node.getDataIndex(index) in index_list:
            assert node.getStdScaling(index) == 1.2345, f"index: {index}"
        else:
            assert node.getStdScaling(index) == 1.0, f"index: {index}"
