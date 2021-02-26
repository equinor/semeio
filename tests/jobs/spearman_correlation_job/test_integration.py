import os
import shutil
import json
import pytest
import pandas as pd
import semeio.workflows.spearman_correlation_job.spearman_correlation as sc
from scipy import stats
from res.enkf import EnKFMain, ResConfig
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)

from unittest.mock import Mock


@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_gen_data(monkeypatch, test_data_root):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scal_job)

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    sc.SpearmanCorrelationJob(ert).run(*["-t", "1.0"])

    # call_args represents the clusters, we expect the snake_oil
    # observations to generate this amount of them
    # call_args is a call object, which itself is a tuple of args and kwargs.
    # In this case, we want args, and the first element of the arguments, which
    # again is a tuple containing the configuration which is a list of configs.
    assert len(list(run_mock.call_args)[0][0]) == 47, "wrong number of clusters"

    cor_matrix_file = os.path.join(
        "storage",
        "snake_oil",
        "reports",
        "default_0",
        "SpearmanCorrelationJob",
        "correlation_matrix.csv",
    )

    pd.read_csv(cor_matrix_file, index_col=[0, 1], header=[0, 1])

    clusters_file = os.path.join(
        "storage",
        "snake_oil",
        "reports",
        "default_0",
        "SpearmanCorrelationJob",
        "clusters.json",
    )
    with open(clusters_file) as f:
        cluster_reports = json.load(f)
        assert len(cluster_reports) == 1

        clusters = cluster_reports[0]
        assert len(clusters.keys()) == 47


@pytest.mark.usefixtures("setup_tmpdir")
def test_scaling(test_data_root):
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    sc.SpearmanCorrelationJob(ert).run(*["-t", "1.0"])

    # assert that this arbitrarily chosen cluster gets scaled as expected
    obs = ert.getObservations()["FOPR"]
    for index in [13, 14, 15, 16, 17, 18, 19, 20]:
        assert obs.getNode(index).getStdScaling() == 2.8284271247461903


@pytest.mark.usefixtures("setup_tmpdir")
def test_skip_clusters_yielding_empty_data_matrixes(monkeypatch, test_data_root):
    def raising_scaling_job(data):
        if data == {"CALCULATE_KEYS": {"keys": [{"index": [88, 89], "key": "FOPR"}]}}:
            raise EmptyDatasetException("foo")

    scaling_mock = Mock(return_value=Mock(**{"run.side_effect": raising_scaling_job}))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scaling_mock)

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    job = sc.SpearmanCorrelationJob(ert)

    try:
        job.run(*["-t", "1.0"])
    except EmptyDatasetException:
        pytest.fail("EmptyDatasetException was not handled by SC job")


@pytest.fixture()
def facade():
    facade = Mock()
    facade.get_observations.return_value = []
    return facade


@pytest.fixture()
def measured_data():
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
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scal_job)

    facade_mock = Mock()
    facade_mock.return_value = facade
    md_mock = Mock()
    md_mock.return_value = measured_data

    monkeypatch.setattr(sc, "LibresFacade", facade_mock)
    monkeypatch.setattr(sc, "MeasuredData", md_mock)

    resconfig_mock = Mock()
    resconfig_mock.model_config.getEnspath.return_value = os.path.join(
        "config_path", "storage_path"
    )

    ert_mock = Mock()
    fs_mock = Mock()
    fs_mock.return_value.getCaseName.return_value = "user_case_name"
    ert_mock.getEnkfFsManager.return_value.getCurrentFileSystem = fs_mock
    ert_mock.resConfig.return_value = resconfig_mock

    sc.SpearmanCorrelationJob(ert_mock).run(*["-t", "1.0"])

    cor_matrix_file = os.path.join(
        "config_path",
        "reports",
        "user_case_name",
        "SpearmanCorrelationJob",
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
        "config_path",
        "reports",
        "user_case_name",
        "SpearmanCorrelationJob",
        "clusters.json",
    )
    with open(clusters_file) as f:
        cluster_reports = json.load(f)
        assert len(cluster_reports) == 1

        clusters = cluster_reports[0]
        assert len(clusters.keys()) == 1
