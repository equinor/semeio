import pytest

from semeio.workflows.localisation.localisation_config import LocalisationConfig

ERT_OBS = ["OBS1", "OBS2"]
ERT_PARAM = {"PARAM_NODE1": ["PARAM1", "PARAM2"]}


def test_simple_config():
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "model_group": {"add": ["PARAM_NODE1:PARAM1"]},
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add[0] == "OBS1"
    assert conf.correlations[0].model_group.add[0] == "PARAM_NODE1:PARAM1"


def test_observations_wilcard():
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS*"]},
                "model_group": {"add": ["PARAM_NODE1:PARAM1"]},
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add == ["OBS1", "OBS2"]
    assert conf.correlations[0].model_group.add[0] == "PARAM_NODE1:PARAM1"


def test_parameters_different_format():
    ERT_PARAM = {
        "PARAM_NODE1": ["PARAM1", "PARAM2"],
        "PARAM_NODE2": ["PARAM1", "PARAM2"],
    }
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "model_group": {
                    "add": [
                        "PARAM_NODE1",
                        {"PARAM_NODE2": ["PARAM1"]},
                    ]
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add[0] == "OBS1"
    assert conf.correlations[0].model_group.add == {
        "PARAM_NODE1": ["PARAM1", "PARAM2"],
        "PARAM_NODE2": ["PARAM1"],
    }


@pytest.parametrize(
    "model_group_add, expected",
    [
        (
            "PARAM_NODE*",
            {
                "PARAM_NODE1": ["PARAM1", "PARAM2"],
                "PARAM_NODE2": ["PARAM1", "PARAM2"],
            },
        ),
        (
            {"PARAM_NODE*": ["PARAM1"]},
            {
                "PARAM_NODE1": ["PARAM1"],
                "PARAM_NODE2": ["PARAM1"],
            },
        ),
        (
            {"PARAM_NODE1": ["PARAM1"]},
            {"PARAM_NODE2": ["PARAM*"]},
            {
                "PARAM_NODE1": ["PARAM1"],
                "PARAM_NODE2": ["PARAM1", "PARAM2"],
            },
        ),
    ],
)
def test_parameters_wildcard(model_group_add, expected):
    ERT_PARAM = {
        "PARAM_NODE1": ["PARAM1", "PARAM2"],
        "PARAM_NODE2": ["PARAM1", "PARAM2"],
    }
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "model_group": {
                    "add": [
                        model_group_add,
                    ]
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add[0] == "OBS1"
    assert conf.correlations[0].model_group.add == expected
