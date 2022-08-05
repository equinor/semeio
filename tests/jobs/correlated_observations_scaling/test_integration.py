# pylint: disable=not-callable
import json
import os
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest
from ert import LibresFacade

from semeio.workflows.correlated_observations_scaling import cos
from semeio.workflows.correlated_observations_scaling.cos import (
    CorrelatedObservationsScalingJob,
)
from tests.jobs.conftest import TEST_DATA_DIR


def get_std_from_obs_vector(vector):
    result = []
    for node in vector:
        for i, _ in enumerate(node):
            result.append(node.getStdScaling(i))
    return result


def test_main_entry_point_gen_data(setup_ert):
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "WPR_DIFF_1"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WPR_DIFF_1", "index": [400, 800]}]},
    }

    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    obs = setup_ert.get_observations()
    obs_vector = obs["WPR_DIFF_1"]

    result = get_std_from_obs_vector(obs_vector)
    assert result == [np.sqrt(4 / 2), np.sqrt(4 / 2), 1.0, 1.0], "Only update subset"

    cos_config["CALCULATE_KEYS"]["keys"][0].update({"index": [400, 800, 1200]})

    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)
    result = get_std_from_obs_vector(obs_vector)
    assert result == [
        np.sqrt(3.0 / 2.0),
        np.sqrt(3.0 / 2.0),
        1.0,
        1.0,
    ], "Change basis for update"

    svd_file = os.path.join(
        "storage",  # ens_path == storage/snake_oil/ensemble
        "snake_oil",
        "reports",
        "snake_oil",
        "default_0",
        "CorrelatedObservationsScalingJob",
        "svd.json",
    )
    # Assert that data was published correctly
    with open(svd_file) as f:
        svd_reports = json.load(f)
        assert len(svd_reports) == 2

        reported_svd = svd_reports[1]
        assert reported_svd == pytest.approx(
            (
                6.531760256452532,
                2.0045135017540487,
                1.1768827000026516,
            ),
            0.1,
        )

    scale_file = os.path.join(
        "storage",  # ens_path == storage/snake_oil/ensemble
        "snake_oil",
        "reports",
        "snake_oil",
        "default_0",
        "CorrelatedObservationsScalingJob",
        "scale_factor.json",
    )
    with open(scale_file) as f:
        scalefactor_reports = json.load(f)
        assert len(scalefactor_reports) == 2

        reported_scalefactor = scalefactor_reports[1]
        assert reported_scalefactor == pytest.approx(1.224744871391589, 0.1)


def test_main_entry_point_summary_data_calc(setup_ert):
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}, {"key": "WOPR_OP1_144"}]}
    }

    obs = setup_ert.get_observations()

    obs_vector = obs["WOPR_OP1_108"]
    result = []
    for index, node in enumerate(obs_vector):
        result.append(node.getStdScaling(index))
    assert result == [1]

    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0


@pytest.mark.parametrize(
    "config, expected_result",
    [
        pytest.param(
            {"CALCULATE_KEYS": {"keys": [{"key": "FOPR"}]}},
            np.sqrt(200 / 5),
            id="All indecies should update",
        ),
        pytest.param(
            {"CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}]}},
            1.0,
            id="No indecies on FOPR should update",
        ),
    ],
)
def test_main_entry_point_history_data_calc(setup_ert, config, expected_result):
    obs = setup_ert.get_observations()
    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, config)
    obs_vector = obs["FOPR"]
    result = []
    for index, node in enumerate(obs_vector):
        result.append(node.getStdScaling(index))
    assert result == [expected_result] * 200


def test_main_entry_point_history_data_calc_subset(setup_ert):
    config = {"CALCULATE_KEYS": {"keys": [{"key": "FOPR", "index": [10, 20]}]}}
    obs = setup_ert.get_observations()
    obs_vector = obs["FOPR"]

    expected_result = [1.0] * 200
    expected_result[10] = np.sqrt(2)
    expected_result[20] = np.sqrt(2)
    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, config)

    result = []
    for index, node in enumerate(obs_vector):
        result.append(node.getStdScaling(index))
    assert (
        result == expected_result
    ), "Check that only the selected subset of obs have updated scaling"


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no equinor libres test-data")
@pytest.mark.equinor_test
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_summary_data_update():
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "WWCT:OP_1"}, {"key": "WWCT:OP_2"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WWCT:OP_2", "index": [1, 2, 3, 4, 5]}]},
    }

    test_data_dir = os.path.join(TEST_DATA_DIR, "Equinor", "config", "obs_testing")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    ert = LibresFacade.from_config_file("config")
    obs = ert.get_observations()
    obs_vector = obs["WWCT:OP_2"]

    ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        if index in cos_config["UPDATE_KEYS"]["keys"][0]["index"]:
            assert node.getStdScaling(index) == np.sqrt(61.0 * 2.0)
        else:
            assert node.getStdScaling(index) == 1.0

    obs_vector = obs["WWCT:OP_1"]

    ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no equinor libres test-data")
@pytest.mark.equinor_test
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_block_data_calc():
    cos_config = {"CALCULATE_KEYS": {"keys": [{"key": "RFT3"}]}}

    test_data_dir = os.path.join(TEST_DATA_DIR, "Equinor", "config", "with_RFT")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    ert = LibresFacade.from_config_file("config")
    obs = ert.get_observations()

    obs_vector = obs["RFT3"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0

    ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 2.0


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no equinor libres test-data")
@pytest.mark.equinor_test
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_block_and_summary_data_calc():
    cos_config = {"CALCULATE_KEYS": {"keys": [{"key": "FOPT"}, {"key": "RFT3"}]}}

    test_data_dir = os.path.join(TEST_DATA_DIR, "Equinor", "config", "with_RFT")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    ert = LibresFacade.from_config_file("config")
    obs = ert.get_observations()

    obs_vector = obs["RFT3"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0

    ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == np.sqrt(64)


def test_main_entry_point_sum_data_update(setup_ert, monkeypatch):
    cos_config = {"CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}]}}

    obs = setup_ert.get_observations()

    obs_vector = obs["WOPR_OP1_108"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0
    monkeypatch.setattr(
        cos.ObservationScaleFactor, "get_scaling_factor", MagicMock(return_value=1.23)
    )
    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.23


def test_main_entry_point_shielded_data(setup_ert, monkeypatch):
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "FOPR", "index": [1, 2, 3, 4, 5]}]}
    }

    obs = setup_ert.get_observations()

    obs_vector = obs["FOPR"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0

    monkeypatch.setattr(
        cos.ObservationScaleFactor, "get_scaling_factor", MagicMock(return_value=1.23)
    )
    setup_ert.run_ertscript(CorrelatedObservationsScalingJob, cos_config)

    for index, node in enumerate(obs_vector):
        if index in [1, 2, 3, 4, 5]:
            assert node.getStdScaling(index) == 1.23, f"index: {index}"
        else:
            assert node.getStdScaling(index) == 1.0, f"index: {index}"
