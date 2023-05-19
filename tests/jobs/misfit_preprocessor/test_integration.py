from unittest.mock import MagicMock, Mock

import pytest
import yaml
from ert.storage import open_storage

import semeio
from semeio.workflows.correlated_observations_scaling import cos
from semeio.workflows.correlated_observations_scaling.exceptions import (
    EmptyDatasetException,
)
from semeio.workflows.misfit_preprocessor import misfit_preprocessor


@pytest.mark.usefixtures("setup_tmpdir")
@pytest.mark.parametrize(
    "observation, expected_nr_clusters", [["*", 45], ["WPR_DIFF_1", 1]]
)
def test_misfit_preprocessor_main_entry_point_gen_data(
    monkeypatch, snake_oil_facade, observation, expected_nr_clusters
):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(
        misfit_preprocessor,
        "CorrelatedObservationsScalingJob",
        scal_job,
    )
    config = {
        "observations": [observation],
        "workflow": {
            "type": "custom_scale",
            "clustering": {"fcluster": {"threshold": 1.0}},
        },
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    with open_storage(snake_oil_facade.enspath, "r") as storage:
        snake_oil_facade.run_ertscript(
            misfit_preprocessor.MisfitPreprocessorJob,
            storage,
            storage.get_ensemble_by_name("default"),
            config_file,
        )

    # call_args represents the clusters, we expect the snake_oil
    # observations to generate this amount of them
    # call_args is a call object, which itself is a tuple of args and kwargs.
    # In this case, we want args, and the first element of the arguments, which
    # again is a tuple containing the configuration which is a list of configs.
    assert (
        len(list(run_mock.call_args)[0][0]) == expected_nr_clusters
    ), "wrong number of clusters"


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_passing_scaling_parameters(monkeypatch, snake_oil_facade):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(
        misfit_preprocessor,
        "CorrelatedObservationsScalingJob",
        scal_job,
    )
    config = {
        "workflow": {
            "type": "custom_scale",
            "pca": {"threshold": 0.5},
        },
    }

    config_file = "my_config_file.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    with open_storage(snake_oil_facade.enspath, "r") as storage:
        snake_oil_facade.run_ertscript(
            misfit_preprocessor.MisfitPreprocessorJob,
            storage,
            storage.get_ensemble_by_name("default"),
            config_file,
        )

    for scaling_config in list(run_mock.call_args)[0][0]:
        assert 0.5 == scaling_config["CALCULATE_KEYS"]["threshold"]


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_main_entry_point_no_config(monkeypatch, snake_oil_facade):
    run_mock = Mock()
    scal_job = Mock(return_value=Mock(run=run_mock))
    monkeypatch.setattr(
        misfit_preprocessor,
        "CorrelatedObservationsScalingJob",
        scal_job,
    )
    with open_storage(snake_oil_facade.enspath, "r") as storage:
        snake_oil_facade.run_ertscript(
            misfit_preprocessor.MisfitPreprocessorJob,
            storage,
            storage.get_ensemble_by_name("default"),
        )

    assert len(run_mock.call_args[0][0]) > 1


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_with_scaling(snake_oil_facade, snapshot):
    config = {
        "workflow": {
            "type": "custom_scale",
            "clustering": {
                "fcluster": {"threshold": 1.0},
            },
        }
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    with open_storage(snake_oil_facade.enspath, "r") as storage:
        snake_oil_facade.run_ertscript(
            misfit_preprocessor.MisfitPreprocessorJob,
            storage,
            storage.get_ensemble_by_name("default"),
            config_file,
        )

    # assert that this arbitrarily chosen cluster gets scaled as expected
    snapshot.assert_match(
        str(
            [
                observation.std_scaling
                for observation in snake_oil_facade.get_observations()["FOPR"]
            ]
        ),
        "cluster_snapshot",
    )


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_skip_clusters_yielding_empty_data_matrixes(
    monkeypatch, snake_oil_facade
):
    # pylint: disable=invalid-name
    def raising_scaling_job(data):
        if data == {"CALCULATE_KEYS": {"keys": [{"index": [88, 89], "key": "FOPR"}]}}:
            raise EmptyDatasetException("foo")

    scaling_mock = Mock(return_value=Mock(**{"run.side_effect": raising_scaling_job}))
    monkeypatch.setattr(
        misfit_preprocessor, "CorrelatedObservationsScalingJob", scaling_mock
    )
    config = {
        "workflow": {
            "type": "custom_scale",
            "clustering": {"fcluster": {"threshold": 1.0}},
        },
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    try:
        with open_storage(snake_oil_facade.enspath, "r") as storage:
            snake_oil_facade.run_ertscript(
                misfit_preprocessor.MisfitPreprocessorJob,
                storage,
                storage.get_ensemble_by_name("default"),
                config_file,
            )
    except EmptyDatasetException:
        pytest.fail("EmptyDatasetException was not handled by misfit preprocessor")


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_invalid_config(snake_oil_facade):
    config = {
        "unknown_key": [],
        "workflow": {"type": "custom_scale", "clustering": {"threshold": 1.0}},
    }
    config_file = "my_config_file.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    expected_err_msg = (
        "Invalid configuration of misfit preprocessor\n"
        "  - extra fields not permitted (workflow.clustering.threshold)\n"
        # There are two clustering functions, and this is invalid in both
        "  - extra fields not permitted (workflow.clustering.threshold)\n"
        "  - extra fields not permitted (unknown_key)\n"
    )
    with pytest.raises(semeio.workflows.misfit_preprocessor.ValidationError) as err:
        with open_storage(snake_oil_facade.enspath, "r") as storage:
            snake_oil_facade.run_ertscript(
                misfit_preprocessor.MisfitPreprocessorJob,
                storage,
                storage.get_ensemble_by_name("default"),
                config_file,
            )
    assert str(err.value) == expected_err_msg


@pytest.mark.usefixtures("setup_tmpdir")
def test_misfit_preprocessor_all_obs(snake_oil_facade, monkeypatch):
    monkeypatch.setattr(
        cos.ObservationScaleFactor, "get_scaling_factor", MagicMock(return_value=1.234)
    )
    with open_storage(snake_oil_facade.enspath, "r") as storage:
        snake_oil_facade.run_ertscript(
            misfit_preprocessor.MisfitPreprocessorJob,
            storage,
            storage.get_ensemble_by_name("default"),
        )

    scaling_factors = []

    obs = snake_oil_facade.get_observations()
    for key in [
        "FOPR",
        "WOPR_OP1_9",
        "WOPR_OP1_36",
        "WOPR_OP1_72",
        "WOPR_OP1_108",
        "WOPR_OP1_144",
        "WOPR_OP1_190",
        "WPR_DIFF_1",
    ]:
        obs_vector = obs[key]
        for index, node in enumerate(obs_vector):
            if isinstance(node.std_scaling, float):
                scaling_factors.append((index, key, node.std_scaling))
            else:
                scaling_factors.append((index, key, node.std_scaling[index]))

    for index, key, scaling_factor in scaling_factors:
        assert scaling_factor == 1.234, f"{index}, {key}"
