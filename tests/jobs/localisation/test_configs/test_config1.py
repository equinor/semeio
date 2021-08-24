from mock import MagicMock

import pytest
import pydantic

from semeio.workflows.localisation.localisation_config import (
    LocalisationConfig,
    expand_wildcards,
    check_for_duplicated_correlation_specifications,
)
from res.enkf.enums.ert_impl_type_enum import ErtImplType


ERT_OBS = ["OBS1", "OBS2", "OBS11", "OBS22", "OBS12", "OBS13", "OBS14", "OBS3"]
ERT_PARAM = [
    "PARAM_NODE1:PARAM1",
    "PARAM_NODE1:PARAM2",
    "PARAM_NODE2:PARAM1",
    "PARAM_NODE2:PARAM2",
    "PARAM_NODE2:PARAM3",
    "PARAM_NODE3:PARAM4",
    "PARAM_NODE22:P1",
    "PARAM_NODE22:P2",
    "PARAM_NODE22:P22",
    "PARAM_NODE1X:X1",
    "PARAM_NODE1X:X2",
    "PARAM_NODE1X:X3",
    "PARAM_NODE2Y:Y1",
    "PARAM_NODE2Y:Y2",
    "PARAM_NODE2Y:Y3",
    "PARAM_FIELD1",
    "PARAM_FIELD2",
    "PARAM_FIELD3",
    "PARAM_GEN:0",
    "PARAM_GEN:1",
    "PARAM_GEN:2",
    "PARAM_GEN:3",
    "PARAM_GEN:4",
    "PARAM_SURFACE1",
    "PARAM_SURFACE2",
]


ERT_NODE_TYPE = {
    "PARAM_NODE1": ErtImplType.GEN_KW,
    "PARAM_NODE2": ErtImplType.GEN_KW,
    "PARAM_NODE3": ErtImplType.GEN_KW,
    "PARAM_NODE22": ErtImplType.GEN_KW,
    "PARAM_NODE1X": ErtImplType.GEN_KW,
    "PARAM_NODE2Y": ErtImplType.GEN_KW,
    "PARAM_FIELD1": ErtImplType.FIELD,
    "PARAM_FIELD2": ErtImplType.FIELD,
    "PARAM_FIELD3": ErtImplType.FIELD,
    "PARAM_GEN": ErtImplType.GEN_DATA,
    "PARAM_SURFACE1": ErtImplType.SURFACE,
    "PARAM_SURFACE2": ErtImplType.SURFACE,
}


@pytest.mark.parametrize(
    "param_group_add,  expected",
    [
        (
            "PARAM_NODE1:*",
            ["PARAM_NODE1:PARAM1", "PARAM_NODE1:PARAM2"],
        ),
        (
            "PARAM_N*1:*",
            ["PARAM_NODE1:PARAM1", "PARAM_NODE1:PARAM2"],
        ),
        (
            ["P*2*", "PARAM_NODE3*"],
            [
                "PARAM_FIELD2",
                "PARAM_GEN:2",
                "PARAM_NODE1:PARAM2",
                "PARAM_NODE1X:X2",
                "PARAM_NODE22:P1",
                "PARAM_NODE22:P2",
                "PARAM_NODE22:P22",
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE2:PARAM2",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE2Y:Y1",
                "PARAM_NODE2Y:Y2",
                "PARAM_NODE2Y:Y3",
                "PARAM_NODE3:PARAM4",
                "PARAM_SURFACE2",
            ],
        ),
        (
            ["P*2:*", "PARAM_NODE3*"],
            [
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE2:PARAM2",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE22:P1",
                "PARAM_NODE22:P2",
                "PARAM_NODE22:P22",
                "PARAM_NODE3:PARAM4",
            ],
        ),
        (
            ["PARAM_NODE3:P*", "PARAM_NODE2*:*2*"],
            [
                "PARAM_NODE2:PARAM2",
                "PARAM_NODE22:P2",
                "PARAM_NODE22:P22",
                "PARAM_NODE2Y:Y2",
                "PARAM_NODE3:PARAM4",
            ],
        ),
        (
            "PARAM_FIELD*",
            ["PARAM_FIELD1", "PARAM_FIELD2", "PARAM_FIELD3"],
        ),
        (
            "PARAM_FIELD1",
            ["PARAM_FIELD1"],
        ),
        (
            "PARAM_GEN:*",
            ["PARAM_GEN:0", "PARAM_GEN:1", "PARAM_GEN:2", "PARAM_GEN:3", "PARAM_GEN:4"],
        ),
    ],
)
def test_simple_config(param_group_add, expected):
    data = {
        "log_level": 2,
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
    assert sorted(conf.correlations[0].param_group.result_items) == sorted(expected)


@pytest.mark.parametrize(
    "obs_group_add, param_group_add,  param_group_remove, expected_error",
    [
        (
            ["OBS*"],
            "PARAM_NODE1:PARAM1",
            "P*:*:*",
            "No match for: P*:*:*",
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
        "log_level": 2,
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
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=expected_error):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


@pytest.mark.parametrize(
    "obsgroup1, paramgroup1,  obsgroup2, paramgroup2, expected_error",
    [
        (
            "OBS1",
            "PARAM_NODE1:*",
            "OBS*",
            "PARAM_NODE*:*",
            "Found 2 duplicated correlations",
        ),
        (
            ["OBS1", "OBS2"],
            ["PARAM_NODE2:*"],
            ["OBS1*", "OBS2*"],
            ["P*:*"],
            "Found 6 duplicated correlations",
        ),
        (
            ["OBS1*"],
            ["PARAM_FIELD2"],
            ["OBS14"],
            ["PARAM_F*"],
            "Found 1 duplicated correlations",
        ),
        (
            "*",
            "*",
            "O*",
            "P*:*",
            "Found 160 duplicated correlations",
        ),
    ],
)
def test_simple_config_duplicate_error(
    obsgroup1, paramgroup1, obsgroup2, paramgroup2, expected_error
):
    data = {
        "log_level": 2,
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
    "ref_point, expected_error",
    [
        (
            [],
            "least 2 items",
        ),
        (
            [100],
            "least 2 items ",
        ),
        (
            ["not_float", 200],
            "value is not a valid float",
        ),
        (
            [100, 200, 300],
            "at most 2 items",
        ),
    ],
)
def test_simple_config_ref_point_error(ref_point, expected_error):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": "OBS",
                },
                "param_group": {
                    "add": "PARAM_NODE1",
                },
                "ref_point": ref_point,
            }
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS"], parameters=["PARAM_NODE1"], **data)


@pytest.mark.parametrize(
    "param_group_add,  param_group_remove, expected",
    [
        (
            "PARAM_NODE1:*",
            "PARAM_NODE2:*",
            ["PARAM_NODE1:PARAM1", "PARAM_NODE1:PARAM2"],
        ),
        (
            "PARAM_N*1*",
            "PARAM_NODE1:PARAM1",
            [
                "PARAM_NODE1:PARAM2",
                "PARAM_NODE1X:X1",
                "PARAM_NODE1X:X2",
                "PARAM_NODE1X:X3",
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE22:P1",
                "PARAM_NODE2Y:Y1",
            ],
        ),
        (
            ["P*2*:*", "PARAM_NODE3*", "P*_GEN*"],
            ["PARAM_NODE2:*", "PARAM_NODE22:P2*", "P*_G*:1", "P*_G*:3", "P*_G*:4"],
            [
                "PARAM_GEN:0",
                "PARAM_GEN:2",
                "PARAM_NODE22:P1",
                "PARAM_NODE2Y:Y1",
                "PARAM_NODE2Y:Y2",
                "PARAM_NODE2Y:Y3",
                "PARAM_NODE3:PARAM4",
            ],
        ),
        (["*FIELD*"], ["*NODE*"], ["PARAM_FIELD1", "PARAM_FIELD2", "PARAM_FIELD3"]),
        (
            ["*"],
            ["PARAM_NODE*"],
            [
                "PARAM_FIELD1",
                "PARAM_FIELD2",
                "PARAM_FIELD3",
                "PARAM_GEN:0",
                "PARAM_GEN:1",
                "PARAM_GEN:2",
                "PARAM_GEN:3",
                "PARAM_GEN:4",
                "PARAM_SURFACE1",
                "PARAM_SURFACE2",
            ],
        ),
    ],
)
def test_add_remove_param_config(param_group_add, param_group_remove, expected):
    data = {
        "log_level": 2,
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
    assert sorted(conf.correlations[0].param_group.result_items) == sorted(expected)


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "add": ["PARAM_NODE1:PARAM1"],
                "remove": "PARAM_NODE*:PARAM1",
            },
            r"Adding: \['PARAM_NODE1:PARAM1'\] and removing: \['PARAM_NODE\*:PARAM1'\]",
        ),
        (
            {
                "add": ["*"],
                "remove": ["*"],
            },
            r"Adding: \['\*'\] and removing: \['\*'\]",
        ),
        (
            {
                "add": ["*FIELD*"],
                "remove": ["*"],
            },
            r"Adding: \['\*FIELD\*'\] and removing: \['\*'\]",
        ),
    ],
)
def test_add_remove_param_config_no_param(config, expected):
    data = {
        "correlations": [
            {"name": "some_name", "obs_group": {"add": ["OBS1"]}, "param_group": config}
        ],
    }
    with pytest.raises(ValueError, match=expected):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


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
        "log_level": 2,
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
    assert conf.correlations[0].obs_group.result_items == expected


@pytest.mark.parametrize(
    "ref_point",
    [
        [550, 1050],
        [100, 150],
        [0, 750],
        [500, 750],
        ["10", 1.0],
    ],
)
def test_ref_point_config(ref_point):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": "OBS",
                },
                "param_group": {
                    "add": "PARAM_NODE1",
                },
                "ref_point": ref_point,
            }
        ],
    }
    conf = LocalisationConfig(observations=["OBS"], parameters=["PARAM_NODE1"], **data)
    expected_refpoint = [float(item) for item in ref_point]
    assert conf.correlations[0].ref_point == expected_refpoint


@pytest.mark.parametrize(
    "pattern, list_of_words, expected_result",
    [(["*"], ["OBS_1", "2"], {"OBS_1", "2"}), ((["OBS*"], ["OBS_1", "2"], {"OBS_1"}))],
)
def test_wildcard_expansion(pattern, list_of_words, expected_result):
    result = expand_wildcards(pattern, list_of_words)
    assert result == expected_result


@pytest.mark.parametrize(
    "pattern, list_of_words, expected_error",
    [
        (["NOT:"], ["OBS_1", "2"], "No match for: NOT"),
        (["OBS", "OBS_1"], ["OBS_1", "2"], "No match for: OBS"),
        (["NOT", "OBS"], ["OBS_1", "2"], "No match for: NOT"),
    ],
)
def test_wildcard_expansion_mismatch(pattern, list_of_words, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        expand_wildcards(pattern, list_of_words)


@pytest.mark.parametrize(
    "obs_1, obs_2, param_1, param_2, expected",
    (
        [["a"], ["a"], ["b"], ["b"], ["Observation: a, parameter: b"]],
        [["a", "c"], ["a"], ["b"], ["b"], ["Observation: a, parameter: b"]],
        [
            ["a", "c"],
            ["a", "c"],
            ["b"],
            ["b"],
            ["Observation: a, parameter: b", "Observation: c, parameter: b"],
        ],
        [
            ["a", "c"],
            ["a", "c"],
            ["b", "d"],
            ["b"],
            ["Observation: a, parameter: b", "Observation: c, parameter: b"],
        ],
    ),
)
def test_check_for_duplicates(obs_1, obs_2, param_1, param_2, expected):
    correlation_1 = MagicMock()
    correlation_2 = MagicMock()
    correlation_1.obs_group.result_items = obs_1
    correlation_1.param_group.result_items = param_1
    correlation_2.obs_group.result_items = obs_2
    correlation_2.param_group.result_items = param_2
    correlations = [correlation_1, correlation_2]
    result = check_for_duplicated_correlation_specifications(correlations)
    assert result == expected
