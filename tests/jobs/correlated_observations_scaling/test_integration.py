import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from ert import LibresFacade
from ert.storage import open_storage

from semeio.workflows.correlated_observations_scaling import cos
from semeio.workflows.correlated_observations_scaling.cos import (
    CorrelatedObservationsScalingJob,
)


def get_std_from_obs_vector(vector):
    result = []
    for node in vector:
        result += list(node.std_scaling)
    return result


def test_main_entry_point_gen_data(snake_oil_facade):
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "WPR_DIFF_1"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WPR_DIFF_1", "index": [400, 800]}]},
    }
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            cos_config,
        )

    obs = snake_oil_facade.get_observations()
    obs_vector = obs["WPR_DIFF_1"]

    result = get_std_from_obs_vector(obs_vector)
    assert result == [np.sqrt(4 / 1), np.sqrt(4 / 1), 1.0, 1.0], "Only update subset"

    cos_config["CALCULATE_KEYS"]["keys"][0].update({"index": [400, 800, 1200]})

    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            cos_config,
        )
    result = get_std_from_obs_vector(obs_vector)
    assert result == [
        np.sqrt(3 / 1),
        np.sqrt(3 / 1),
        1.0,
        1.0,
    ], "Change basis for update"

    svd_file = os.path.join(
        "storage",  # ens_path == storage/snake_oil/ensemble
        "snake_oil",
        "reports",
        "snake_oil",
        "default",
        "CorrelatedObservationsScalingJob",
        "svd.json",
    )
    # Assert that data was published correctly
    with open(svd_file, encoding="utf-8") as file:
        svd_reports = json.load(file)
        assert len(svd_reports) == 2

        reported_svd = svd_reports[1]
        assert reported_svd == pytest.approx(
            (4.414283609038025, 1.9625780125365313, 0.74244484701782),
            0.1,
        )

    scale_file = os.path.join(
        "storage",  # ens_path == storage/snake_oil/ensemble
        "snake_oil",
        "reports",
        "snake_oil",
        "default",
        "CorrelatedObservationsScalingJob",
        "scale_factor.json",
    )
    with open(scale_file, encoding="utf-8") as file:
        scalefactor_reports = json.load(file)
        assert len(scalefactor_reports) == 2

        reported_scalefactor = scalefactor_reports[1]
        assert reported_scalefactor == pytest.approx(np.sqrt(3 / 1), 0.1)


def test_main_entry_point_summary_data_calc(snake_oil_facade):
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}, {"key": "WOPR_OP1_144"}]}
    }

    obs = snake_oil_facade.get_observations()

    obs_vector = obs["WOPR_OP1_108"]
    assert [node.std_scaling for node in obs_vector] == [1]

    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            cos_config,
        )

    for node in obs_vector:
        assert node.std_scaling == np.sqrt(2 / 1)


@pytest.mark.parametrize(
    "config, expected_result",
    [
        pytest.param(
            {"CALCULATE_KEYS": {"keys": [{"key": "FOPR"}]}},
            np.sqrt(200 / 3),
            id="All indecies should update",
        ),
        pytest.param(
            {"CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}]}},
            1.0,
            id="No indices on FOPR should update",
        ),
    ],
)
def test_main_entry_point_history_data_calc(snake_oil_facade, config, expected_result):
    obs = snake_oil_facade.get_observations()
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            config,
        )
    obs_vector = obs["FOPR"]
    assert [node.std_scaling for node in obs_vector] == [expected_result] * 200


def test_main_entry_point_history_data_calc_subset(snake_oil_facade):
    config = {"CALCULATE_KEYS": {"keys": [{"key": "FOPR", "index": [10, 20]}]}}
    obs = snake_oil_facade.get_observations()
    obs_vector = obs["FOPR"]

    expected_result = [1.0] * 200
    expected_result[10] = np.sqrt(2)
    expected_result[20] = np.sqrt(2)
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            config,
        )

    assert [
        node.std_scaling for node in obs_vector
    ] == expected_result, (
        "Check that only the selected subset of obs have updated scaling"
    )


def test_main_entry_point_sum_data_update(snake_oil_facade, monkeypatch):
    cos_config = {"CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}]}}

    obs = snake_oil_facade.get_observations()

    obs_vector = obs["WOPR_OP1_108"]

    for node in obs_vector:
        assert node.std_scaling == 1.0
    monkeypatch.setattr(
        cos.ObservationScaleFactor, "get_scaling_factor", MagicMock(return_value=1.23)
    )
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            cos_config,
        )

    for node in obs_vector:
        assert node.std_scaling == 1.23


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_main_entry_point_shielded_data(monkeypatch):
    ert = LibresFacade.from_config_file("snake_oil.ert")
    cos_config = {
        "CALCULATE_KEYS": {"keys": [{"key": "FOPR", "index": [1, 2, 3, 4, 5]}]}
    }

    obs = ert.get_observations()

    obs_vector = obs["FOPR"]

    for node in obs_vector:
        assert node.std_scaling == 1.0

    monkeypatch.setattr(
        cos.ObservationScaleFactor, "get_scaling_factor", MagicMock(return_value=1.23)
    )
    with open_storage(ert.enspath, "w") as storage:
        ert.run_ertscript(
            CorrelatedObservationsScalingJob,
            storage,
            storage.get_ensemble_by_name("default"),
            cos_config,
        )

    for index, node in enumerate(obs_vector):
        if index in [1, 2, 3, 4, 5]:
            assert node.std_scaling == 1.23
        else:
            assert node.std_scaling == 1.0
