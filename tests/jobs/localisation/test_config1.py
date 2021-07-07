import pytest

# from pydantic import ValidationError

from semeio.workflows.localisation.localisation_config import LocalisationConfig

# from res.enkf.enums.ert_impl_type_enum import ErtImplType

ERT_GRID_CONFIG = {
    "dimensions": [100, 200],
    "grid_origo": [100.0, 150.0],
    "rotation": 45,
    "size": [1000, 2000],
}


ERT_OBS = ["OBS1", "OBS2", "OBS11", "OBS22", "OBS12", "OBS13", "OBS14", "OBS3"]
ERT_PARAM = {
    "PARAM_NODE1": ["PARAM1", "PARAM2"],
    "PARAM_NODE2": ["PARAM1", "PARAM2", "PARAM3"],
    "PARAM_NODE3": ["PARAM4"],
    "PARAM_NODE22": ["P1", "P2", "P22"],
    "PARAM_NODE1X": ["X1", "X2", "X3"],
    "PARAM_NODE2Y": ["Y1", "Y2", "Y3"],
    "PARAM_FIELD1": [],
    "PARAM_FIELD2": [],
    "PARAM_FIELD3": [],
    "PARAM_GEN": [],
}


@pytest.mark.parametrize(
    "param_group_add,  expected",
    [
        (
            "PARAM_NODE1:*",
            {"PARAM_NODE1": ["PARAM1", "PARAM2"]},
        ),
        (
            "PARAM_N*1:*",
            {
                "PARAM_NODE1": ["PARAM1", "PARAM2"],
            },
        ),
        (
            ["PARAM_NODE1:*", "PARAM_NODE2:*"],
            {
                "PARAM_NODE1": ["PARAM1", "PARAM2"],
                "PARAM_NODE2": ["PARAM1", "PARAM2", "PARAM3"],
            },
        ),
        (
            ["PARAM_NODE2:*", "PARAM_NODE1:*"],
            {
                "PARAM_NODE1": ["PARAM1", "PARAM2"],
                "PARAM_NODE2": ["PARAM1", "PARAM2", "PARAM3"],
            },
        ),
        (
            ["P*2*", "PARAM_NODE3*"],
            {
                "PARAM_NODE1": ["PARAM2"],
                "PARAM_NODE1X": ["X2"],
                "PARAM_NODE2": ["PARAM1", "PARAM2", "PARAM3"],
                "PARAM_NODE22": ["P1", "P2", "P22"],
                "PARAM_NODE2Y": ["Y1", "Y2", "Y3"],
                "PARAM_NODE3": ["PARAM4"],
                "PARAM_FIELD2": [],
            },
        ),
        (
            ["P*2:*", "PARAM_NODE3*"],
            {
                "PARAM_NODE2": ["PARAM1", "PARAM2", "PARAM3"],
                "PARAM_NODE22": ["P1", "P2", "P22"],
                "PARAM_NODE3": ["PARAM4"],
            },
        ),
        (
            "PARAM_NODE1:PARAM1",
            {"PARAM_NODE1": ["PARAM1"]},
        ),
        (
            [
                "PARAM_NODE1:PARAM1",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE3:PARAM4",
                "PARAM_NODE1X:X*",
            ],
            {
                "PARAM_NODE1": ["PARAM1"],
                "PARAM_NODE2": ["PARAM3"],
                "PARAM_NODE3": ["PARAM4"],
                "PARAM_NODE1X": ["X1", "X2", "X3"],
            },
        ),
        (
            ["PARAM_NODE3:P*", "PARAM_NODE2*:*2*"],
            {
                "PARAM_NODE2": ["PARAM2"],
                "PARAM_NODE22": ["P2", "P22"],
                "PARAM_NODE2Y": ["Y2"],
                "PARAM_NODE3": ["PARAM4"],
            },
        ),
        (
            [
                "PARAM_NODE1:PARAM1",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE2Y:*",
                "PARAM_NODE1X:X3",
                "PARAM_NODE1X:X1",
            ],
            {
                "PARAM_NODE1": ["PARAM1"],
                "PARAM_NODE1X": ["X1", "X3"],
                "PARAM_NODE2": ["PARAM3"],
                "PARAM_NODE2Y": ["Y1", "Y2", "Y3"],
            },
        ),
        (
            "PARAM_FIELD*",
            {
                "PARAM_FIELD1": [],
                "PARAM_FIELD2": [],
                "PARAM_FIELD3": [],
            },
        ),
        (
            "PARAM_FIELD1",
            {
                "PARAM_FIELD1": [],
            },
        ),
    ],
)
def test_simple_config(param_group_add, expected):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "param_group": {
                    "add": param_group_add,
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add[0] == "OBS1"
    assert conf.correlations[0].param_group.add == expected


@pytest.mark.parametrize(
    "obs_group_add, param_group_add,  param_group_remove, expected_error",
    [
        (
            "OBS1",
            "PARAM_NODE1",
            "[]",
            "List of specified model parameters",
        ),
        (
            ["OBS1", "OBS2"],
            ["PY*:*", "PARAM_FILED*"],
            "[]",
            "List of specified model parameters",
        ),
        (
            ["OBS2*"],
            ["PARAM_FIELD*"],
            ["PARAM_FIELD2:*"],
            "List of specified model parameters",
        ),
        (
            ["OBS*"],
            "PARAM_NODE1:PARAM1",
            "P*:*:*",
            "List of specified model parameters",
        ),
        (
            [],
            "PARAM_NODE1:PARAM1",
            [],
            "",
        ),
    ],
)
def test_simple_config_error(
    obs_group_add, param_group_add, param_group_remove, expected_error
):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                },
                "param_group": {
                    "add": param_group_add,
                    "remove": param_group_remove,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


@pytest.mark.parametrize(
    "obsgroup1, paramgroup1,  obsgroup2, paramgroup2, expected_error",
    [
        (
            "OBS1",
            "PARAM_NODE1:*",
            "OBS*",
            "PARAM_NODE*:*",
            "Some correlations are specified multiple times. ",
        ),
        (
            ["OBS1", "OBS2"],
            ["PARAM_NODE2:*"],
            ["OBS1*", "OBS2*"],
            ["P*:*"],
            "Some correlations are specified multiple times. ",
        ),
        (
            ["OBS1*"],
            ["PARAM_FIELD2"],
            ["OBS14"],
            ["PARAM_F*"],
            "Some correlations are specified multiple times. ",
        ),
        (
            "*",
            "*",
            "O*",
            "P*:*",
            "Some correlations are specified multiple times. ",
        ),
    ],
)
def test_simple_config_duplicate_error(
    obsgroup1, paramgroup1, obsgroup2, paramgroup2, expected_error
):
    data = {
        "correlations": [
            {
                "name": "some_name1",
                "obs_group": {
                    "add": obsgroup1,
                },
                "param_group": {
                    "add": paramgroup1,
                },
            },
            {
                "name": "some_name2",
                "obs_group": {
                    "add": obsgroup2,
                },
                "param_group": {
                    "add": paramgroup2,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


@pytest.mark.parametrize(
    "obs_group_add, obs_group_remove, ref_point, expected_error",
    [
        (
            "OBS1*",
            "OBS1",
            [],
            "Reference point for an observation group must be a list of ",
        ),
        (
            "OBS1*",
            "OBS1",
            [100],
            "Reference point for an observation group must be a list of ",
        ),
        (
            "OBS1*",
            "OBS1",
            ["1x00", 200],
            "could not convert string to float",
        ),
        (
            "OBS1*",
            "OBS1",
            [100, 200, 300],
            "Reference point for an observation group must be a list of ",
        ),
    ],
)
def test_simple_config_ref_point_error(
    obs_group_add, obs_group_remove, ref_point, expected_error
):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                    "remove": obs_group_remove,
                },
                "param_group": {
                    "add": "PARAM_NODE1:PARAM1",
                },
                "ref_point": ref_point,
            }
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(
            observations=ERT_OBS,
            parameters=ERT_PARAM,
            grid_config=ERT_GRID_CONFIG,
            **data
        )


@pytest.mark.parametrize(
    "param_group_add,  param_group_remove, expected",
    [
        (
            "PARAM_NODE1:*",
            "PARAM_NODE2:*",
            {"PARAM_NODE1": ["PARAM1", "PARAM2"]},
        ),
        (
            "PARAM_N*1*",
            "PARAM_NODE1:PARAM1",
            {
                "PARAM_NODE1": ["PARAM2"],
                "PARAM_NODE1X": ["X1", "X2", "X3"],
                "PARAM_NODE2": ["PARAM1"],
                "PARAM_NODE22": ["P1"],
                "PARAM_NODE2Y": ["Y1"],
            },
        ),
        (
            ["PARAM_NODE1:*", "PARAM_NODE2:*"],
            ["PARAM_NODE1:PARAM1", "PARAM_NODE2:PARAM1"],
            {
                "PARAM_NODE1": ["PARAM2"],
                "PARAM_NODE2": ["PARAM2", "PARAM3"],
            },
        ),
        (
            ["PARAM_NODE2*", "PARAM_NODE1:PARAM2"],
            ["PARAM_NODE2:PARAM2", "PARAM_NODE2Y:Y2"],
            {
                "PARAM_NODE1": ["PARAM2"],
                "PARAM_NODE2": ["PARAM1", "PARAM3"],
                "PARAM_NODE22": ["P1", "P2", "P22"],
                "PARAM_NODE2Y": ["Y1", "Y3"],
            },
        ),
        (
            ["P*2*", "PARAM_NODE3*"],
            ["PARAM_NODE2:*", "PARAM_NODE22:P2*"],
            {
                "PARAM_FIELD2": [],
                "PARAM_NODE1": ["PARAM2"],
                "PARAM_NODE1X": ["X2"],
                "PARAM_NODE22": ["P1"],
                "PARAM_NODE2Y": ["Y1", "Y2", "Y3"],
                "PARAM_NODE3": ["PARAM4"],
            },
        ),
        (
            ["PARAM_NODE1:PARAM1"],
            "PARAM_NODE*:PARAM1",
            {},
        ),
        (
            [
                "PARAM_NODE1:PARAM1",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE1X:X2",
                "PARAM_NODE1X:X1",
            ],
            ["P*_N*X:X2", "PARAM_NODE2:PARAM1"],
            {
                "PARAM_NODE1": ["PARAM1"],
                "PARAM_NODE2": ["PARAM3"],
                "PARAM_NODE1X": ["X1"],
            },
        ),
        (
            [
                "PARAM_NODE1:PARAM1",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE1X:X2",
                "PARAM_NODE1X:X1",
            ],
            ["PARAM_NODE*Y:Y*"],
            {
                "PARAM_NODE1": ["PARAM1"],
                "PARAM_NODE1X": ["X1", "X2"],
                "PARAM_NODE2": ["PARAM1", "PARAM3"],
            },
        ),
        (
            [
                "PARAM_NODE1:PARAM1",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE2Y:*",
                "PARAM_NODE1X:X3",
                "PARAM_NODE1X:X1",
                "PARAM_NODE1:PARAM2",
            ],
            [
                "PARAM_NODE2Y:Y3",
                "PARAM_NODE1:PARAM1",
            ],
            {
                "PARAM_NODE1": ["PARAM2"],
                "PARAM_NODE1X": ["X1", "X3"],
                "PARAM_NODE2": ["PARAM3"],
                "PARAM_NODE2Y": ["Y1", "Y2"],
            },
        ),
        (
            ["*"],
            ["*"],
            {},
        ),
        (
            ["*FIELD*"],
            ["*"],
            {},
        ),
        (
            ["*FIELD*"],
            ["*NODE*"],
            {
                "PARAM_FIELD1": [],
                "PARAM_FIELD2": [],
                "PARAM_FIELD3": [],
            },
        ),
        (
            ["*"],
            ["PARAM_NODE*"],
            {
                "PARAM_FIELD1": [],
                "PARAM_FIELD2": [],
                "PARAM_FIELD3": [],
                "PARAM_GEN": [],
            },
        ),
    ],
)
def test_add_remove_param_config(param_group_add, param_group_remove, expected):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "param_group": {
                    "add": param_group_add,
                    "remove": param_group_remove,
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add[0] == "OBS1"
    assert conf.correlations[0].param_group.add == expected


@pytest.mark.parametrize(
    "obs_group_add, obs_group_remove, expected",
    [
        (
            "OBS*",
            "OBS*2",
            ["OBS1", "OBS11", "OBS13", "OBS14", "OBS3"],
        ),
        (
            "*",
            "OBS*2",
            ["OBS1", "OBS11", "OBS13", "OBS14", "OBS3"],
        ),
        (
            "*2*",
            "*1*",
            ["OBS2", "OBS22"],
        ),
        (
            "*3",
            "*1*",
            ["OBS3"],
        ),
    ],
)
def test_add_remove_obs_config(obs_group_add, obs_group_remove, expected):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": [obs_group_add], "remove": [obs_group_remove]},
                "param_group": {
                    "add": ["PARAM_NODE1:PARAM1"],
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add == expected
    assert conf.correlations[0].param_group.add == {"PARAM_NODE1": ["PARAM1"]}


@pytest.mark.parametrize(
    "obs_group_add, param_group_add, ref_point,  expected",
    [
        (
            "OBS1*",
            "PARAM_FIELD1",
            [550, 1050],
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [100, 150],
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [0, 750],
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [500, 750],
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
    ],
)
def test_add_remove_obs_with_ref_point_config(
    obs_group_add, param_group_add, ref_point, expected
):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": [obs_group_add],
                },
                "param_group": {
                    "add": [param_group_add],
                },
                "ref_point": ref_point,
            }
        ],
    }
    conf = LocalisationConfig(
        observations=ERT_OBS, parameters=ERT_PARAM, grid_config=ERT_GRID_CONFIG, **data
    )
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add == expected
    assert conf.correlations[0].param_group.add == {"PARAM_FIELD1": []}


@pytest.mark.parametrize(
    "obs_add, param_add, ref_point,  method, range1, range2, angle, expected",
    [
        (
            "OBS1*",
            "PARAM_FIELD1",
            [550, 1050],
            "gaussian_decay",
            1700,
            850,
            310,
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [100, 150],
            "exponential_decay",
            0.1,
            850,
            0,
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [0, 750],
            "exponential_decay",
            700,
            850,
            100,
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [500, 750],
            "exponential_decay",
            1700,
            500,
            360,
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
        (
            "OBS1*",
            "PARAM_FIELD1",
            [500, 750],
            "gaussian_decay",
            0.1,
            0.1,
            0,
            ["OBS1", "OBS11", "OBS12", "OBS13", "OBS14"],
        ),
    ],
)
def test_add_remove_obs_with_ref_point_and_field_scale_config(
    obs_add, param_add, ref_point, method, range1, range2, angle, expected
):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": [obs_add],
                },
                "param_group": {
                    "add": [param_add],
                },
                "ref_point": ref_point,
                "field_scale": {
                    "method": method,
                    "main_range": range1,
                    "perp_range": range2,
                    "angle": angle,
                },
            }
        ],
    }
    conf = LocalisationConfig(
        observations=ERT_OBS, parameters=ERT_PARAM, grid_config=ERT_GRID_CONFIG, **data
    )
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.add == expected
    assert conf.correlations[0].param_group.add == {"PARAM_FIELD1": []}
