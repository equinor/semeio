import os
import shutil
import sys

import pytest

import semeio.jobs.scripts.spearman_correlation as sc
from res.enkf import EnKFMain, ResConfig
from semeio.jobs.correlated_observations_scaling.exceptions import EmptyDatasetException
from tests.jobs.correlated_observations_scaling.conftest import TEST_DATA_DIR

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no libres test-data")
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_gen_data(monkeypatch):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scal_job)

    test_data_dir = os.path.join(TEST_DATA_DIR, "local", "snake_oil")

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
    assert len(run_mock.call_args[0][0]) == 47, "wrong number of clusters"


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no libres test-data")
@pytest.mark.usefixtures("setup_tmpdir")
def test_scaling():
    test_data_dir = os.path.join(TEST_DATA_DIR, "local", "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    sc.SpearmanCorrelationJob(ert).run(*["-t", "1.0"])

    # assert that this arbitrarily chosen cluster gets scaled as expected
    obs = ert.getObservations()["FOPR"]
    for index in [13, 14, 15, 16, 17, 18, 19, 20]:
        assert obs.getNode(index).getStdScaling() == 2.8284271247461903


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no libres test-data")
@pytest.mark.usefixtures("setup_tmpdir")
def test_skip_clusters_yielding_empty_data_matrixes(monkeypatch):
    def raising_scaling_job(data):
        if data == {"CALCULATE_KEYS": {"keys": [{"index": [88, 89], "key": "FOPR"}]}}:
            raise EmptyDatasetException("foo")

    scaling_mock = Mock(return_value=Mock(**{"run.side_effect": raising_scaling_job}))
    monkeypatch.setattr(sc, "CorrelatedObservationsScalingJob", scaling_mock)

    test_data_dir = os.path.join(TEST_DATA_DIR, "local", "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)
    job = sc.SpearmanCorrelationJob(ert)

    try:
        job.run(*["-t", "1.0"])
    except EmptyDatasetException:
        pytest.fail("EmptyDatasetException was not handled by SC job")
