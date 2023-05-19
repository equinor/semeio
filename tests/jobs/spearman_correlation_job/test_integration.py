# pylint: disable=not-callable
import json
import os
from unittest.mock import Mock

import pandas as pd
import pytest
from ert.storage import open_storage
from scipy import stats

import semeio.workflows.spearman_correlation_job.spearman_correlation as sc
from semeio.communication import semeio_script
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)


@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_gen_data(monkeypatch, snake_oil_facade):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scal_job)

    facade = snake_oil_facade
    with open_storage(facade.enspath, "w") as storage:
        output_dir = facade.run_ertscript(
            sc.SpearmanCorrelationJob,
            storage,
            storage.get_ensemble_by_name("default"),
            *["-t", "1.0"],
        )

    # call_args represents the clusters, we expect the snake_oil
    # observations to generate this amount of them
    # call_args is a call object, which itself is a tuple of args and kwargs.
    # In this case, we want args, and the first element of the arguments, which
    # again is a tuple containing the configuration which is a list of configs.
    assert len(list(run_mock.call_args)[0][0]) == 43, "wrong number of clusters"

    cor_matrix_file = os.path.join(
        output_dir,
        "correlation_matrix.csv",
    )

    pd.read_csv(cor_matrix_file, index_col=[0, 1], header=[0, 1])

    clusters_file = os.path.join(
        output_dir,
        "clusters.json",
    )
    with open(clusters_file, encoding="utf-8") as file:
        cluster_reports = json.load(file)
        assert len(cluster_reports) == 1

        clusters = cluster_reports[0]
        assert len(clusters.keys()) == 43


def test_scaling(snake_oil_facade, snapshot):
    facade = snake_oil_facade

    with open_storage(facade.enspath, "w") as storage:
        facade.run_ertscript(
            sc.SpearmanCorrelationJob,
            storage,
            storage.get_ensemble_by_name("default"),
            *["-t", "1.0"],
        )

    # assert that this arbitrarily chosen cluster gets scaled as expected
    snapshot.assert_match(
        str(
            [
                observation.std_scaling
                for observation in facade.get_observations()["FOPR"]
            ]
        ),
        "cluster_snapshot",
    )


def test_skip_clusters_yielding_empty_data_matrixes(monkeypatch, snake_oil_facade):
    def raising_scaling_job(data):
        if data == {"CALCULATE_KEYS": {"keys": [{"index": [88, 89], "key": "FOPR"}]}}:
            raise EmptyDatasetException("foo")

    scaling_mock = Mock(return_value=Mock(**{"run.side_effect": raising_scaling_job}))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scaling_mock)

    facade = snake_oil_facade

    try:
        with open_storage(facade.enspath, "w") as storage:
            facade.run_ertscript(
                sc.SpearmanCorrelationJob,
                storage,
                storage.get_ensemble_by_name("default"),
                *["-t", "1.0"],
            )
    except EmptyDatasetException:
        pytest.fail("EmptyDatasetException was not handled by SC job")


@pytest.fixture(name="facade")
def fixture_facade():
    facade = Mock()
    facade.get_observations.return_value.obs_vectors = {}
    return facade


@pytest.fixture(name="measured_data")
def fixture_measured_data():
    # pylint: disable=invalid-name
    r1 = [1, 5, 3]
    r2 = [2, 90, 2]
    r3 = [3, 2, 1]
    measured_data = Mock()
    columns = [("obs_name", n) for n in range(3)]
    data = pd.DataFrame(
        [r1, r2, r3],
        columns=pd.MultiIndex.from_tuples(columns, names=["key_index", "data_index"]),
    )
    measured_data.get_simulated_data.return_value = data
    return measured_data


@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_syn_data(monkeypatch, facade, measured_data):
    # pylint: disable=too-many-locals,invalid-name
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    facade_mock = Mock()
    facade_mock.return_value = facade
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scal_job)
    monkeypatch.setattr(semeio_script, "LibresFacade", facade_mock)

    facade.get_measured_data.return_value = measured_data

    facade.enspath = os.path.join("config_path", "storage_path")
    facade.user_config_file = "config_name"

    ert_mock = Mock()
    fs_mock = Mock()
    fs_mock.return_value.getCaseName.return_value = "user_case_name"
    ert_mock.getEnkfFsManager.return_value.getCurrentFileSystem = fs_mock
    ensemble_mock = Mock()
    ensemble_mock.name = "default"
    job = sc.SpearmanCorrelationJob(ert_mock, Mock(), ensemble_mock)
    job.run(*["-t", "1.0"])
    cor_matrix_file = os.path.join(
        # pylint: disable=protected-access
        job._output_dir,
        "correlation_matrix.csv",
    )

    r1 = [1, 5, 3]
    r2 = [2, 90, 2]
    r3 = [3, 2, 1]
    expected_corr_matrix = []
    for a in zip(r1, r2, r3):
        row = []
        for b in zip(r1, r2, r3):
            expected_corr = stats.spearmanr(a, b)
            row.append(expected_corr[0])
        expected_corr_matrix.append(row)

    corr_matrix = pd.read_csv(cor_matrix_file, index_col=[0, 1], header=[0, 1])

    assert (expected_corr_matrix == corr_matrix.values).all()
    clusters_file = os.path.join(
        # pylint: disable=protected-access
        job._output_dir,
        "clusters.json",
    )
    with open(clusters_file, encoding="utf-8") as file:
        cluster_reports = json.load(file)
        assert len(cluster_reports) == 1

        clusters = cluster_reports[0]
        assert len(clusters.keys()) == 1
