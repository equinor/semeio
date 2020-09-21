import os
import shutil
import yaml
import pytest

from res.enkf import EnKFMain, ResConfig

import semeio
import semeio.workflows.misfit_preprocessor.misfit_preprocessor as misfit_preprocessor
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)

from unittest.mock import Mock


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_main_entry_point_gen_data(monkeypatch, test_data_root):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(
        misfit_preprocessor,
        "CorrelatedObservationsScalingJob",
        scal_job,
    )

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    config = {
        "clustering": {
            "method": "spearman_correlation",
            "spearman_correlation": {"fcluster": {"t": 1.0}},
        }
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    misfit_preprocessor.MisfitPreprocessorJob(ert).run(config_file)

    # call_args represents the clusters, we expect the snake_oil
    # observations to generate this amount of them
    # call_args is a call object, which itself is a tuple of args and kwargs.
    # In this case, we want args, and the first element of the arguments, which
    # again is a tuple containing the configuration which is a list of configs.
    assert len(list(run_mock.call_args)[0][0]) == 58, "wrong number of clusters"


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_passing_scaling_parameters(monkeypatch, test_data_root):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(
        misfit_preprocessor,
        "CorrelatedObservationsScalingJob",
        scal_job,
    )

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    config = {
        "clustering": {"method": "spearman_correlation"},
        "scaling": {"threshold": 0.5, "std_cutoff": 2, "alpha": 3},
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    misfit_preprocessor.MisfitPreprocessorJob(ert).run(config_file)

    for scaling_config in list(run_mock.call_args)[0][0]:
        assert 0.5 == scaling_config["CALCULATE_KEYS"]["threshold"]


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_main_entry_point_no_config(monkeypatch, test_data_root):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(
        misfit_preprocessor,
        "CorrelatedObservationsScalingJob",
        scal_job,
    )

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    misfit_preprocessor.MisfitPreprocessorJob(ert).run()

    assert len(run_mock.call_args[0][0]) > 1  # pylint: disable=unsubscriptable-object


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_with_scaling(test_data_root):
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    config = {
        "clustering": {
            "method": "spearman_correlation",
            "spearman_correlation": {"fcluster": {"t": 1.0}},
        }
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    misfit_preprocessor.MisfitPreprocessorJob(ert).run(config_file)

    # assert that this arbitrarily chosen cluster gets scaled as expected
    obs = ert.getObservations()["FOPR"]
    for index in [13, 14, 15, 16, 17, 18, 19, 20]:
        assert obs.getNode(index).getStdScaling() == 2.8284271247461903


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_skip_clusters_yielding_empty_data_matrixes(
    monkeypatch, test_data_root
):
    def raising_scaling_job(data):
        if data == {"CALCULATE_KEYS": {"keys": [{"index": [88, 89], "key": "FOPR"}]}}:
            raise EmptyDatasetException("foo")

    scaling_mock = Mock(return_value=Mock(**{"run.side_effect": raising_scaling_job}))
    monkeypatch.setattr(
        misfit_preprocessor, "CorrelatedObservationsScalingJob", scaling_mock
    )

    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    config = {
        "clustering": {
            "method": "spearman_correlation",
            "spearman_correlation": {"fcluster": {"t": 1.0}},
        }
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    job = misfit_preprocessor.MisfitPreprocessorJob(ert)

    try:
        job.run(config_file)
    except EmptyDatasetException:
        pytest.fail("EmptyDatasetException was not handled by misfit preprocessor")


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_invalid_config(test_data_root):
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    config = {
        "unknown_key": [],
        "clustering": {
            "method": "spearman_correlation",
            "spearman_correlation": {"fcluster": {"threshold": 1.0}},
        },
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    job = misfit_preprocessor.MisfitPreprocessorJob(ert)
    with pytest.raises(semeio.workflows.misfit_preprocessor.ValidationError) as ve:
        job.run(config_file)

    expected_err_msg = (
        "Invalid configuration of misfit preprocessor\n"
        "  - Unknown key: unknown_key (root level)\n"
        "  - Unknown key: threshold (clustering.spearman_correlation.fcluster)\n"
    )
    assert expected_err_msg == str(ve.value)
