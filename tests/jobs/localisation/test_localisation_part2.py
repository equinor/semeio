from typing import List
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

import semeio.workflows.localisation.local_script_lib as local
from semeio.workflows.localisation.localisation_config import (
    LocalisationConfig,
    CorrelationConfig,
)


def test_read_correlation_specification():
    local.debug_print("\n\nRun: test_read_correlation_specification")
    ert_list_all_obs = [
        "OP_1_WWCT1",
        "OP_1_WWCT2",
        "OP_1_WWCT3",
        "OP_2_WWCT1",
        "OP_2_WWCT2",
        "OP_2_WWCT3",
        "OP_1_WGORT1",
        "OP_1_WGORT2",
        "OP_1_WGORT3",
        "OP_2_WGORT1",
        "OP_2_WGORT2",
        "OP_2_WGORT3",
        "ROP_1_OBS",
        "SHUT_IN_OP1",
    ]
    ert_param_dict = {
        "INTERPOLATE_RELPERM": ["INTERPOLATE_WO", "INTERPOLATE_GO"],
        "MULTFLT": [
            "MULTFLT_F1",
            "MULTFLT_F2",
            "MULTFLT_F3",
            "MULTFLT_F4",
            "MULTFLT_F5",
        ],
        "MULTZ": ["MULTZ_MIDREEK", "MULTZ_LOWREEK"],
        "RMSGLOBPARAMS": ["FWL", "COHIBA_MODEL_MODE"],
        "GENPARAM1": None,
        "GENPARAM11": None,
        "GENPARAM12": None,
        "GENPARAM2": None,
        "GENPARAM21": None,
        "FIELDPARAM1": None,
    }

    obs_group_dict = {
        "OP_1_GROUP": ["OP_1_WWCT1", "OP_1_WWCT2", "OP_1_WWCT3"],
        "OP_2_GROUP": ["OP_2_WWCT1", "OP_2_WWCT2", "OP_2_WWCT3"],
    }
    param_group_dict = {
        "MULTZ_GROUP": ["MULTZ:MULTZ_LOWREEK", "MULTZ:MULTZ_MIDREEK"],
        "MULTFLT_GROUP1": [
            "MULTFLT:MULTFLT_F1",
            "MULTFLT:MULTFLT_F2",
            "MULTFLT:MULTFLT_F3",
        ],
        "MULTFLT_GROUP2": [
            "MULTFLT:MULTFLT_F1",
            "MULTFLT:MULTFLT_F2",
            "MULTFLT:MULTFLT_F3",
            "MULTFLT:MULTFLT_F4",
            "MULTFLT:MULTFLT_F5",
        ],
        "RMSGLOBPARAMS_GROUP": ["RMSGLOBPARAMS:FWL"],
        "INTERPOLATE_RELPERM_GROUP": [
            "INTERPOLATE_RELPERM:INTERPOLATE_GO",
            "INTERPOLATE_RELPERM:INTERPOLATE_WO",
        ],
        "GENPARAM_GROUP": ["GENPARAM1", "GENPARAM21"],
    }
    # Test 1:
    all_kw = {
        "correlations": [
            {
                "name": "CORRELATION1",
                "obs_group": {"add": ["OP_*_WWCT1"]},
                "model_group": {"add": ["MULTZ"]},
            },
            {
                "name": "CORRELATION2",
                "obs_group": {"add": "All", "remove": ["ROP_*_OBS", "SHUT_IN*"]},
                "model_group": {
                    "add": "All",
                    "remove": [
                        "MULTFLT:MULTFLT_F1",
                        "MULTFLT:MULTFLT_F2",
                        "MULTZ",
                        "INTERPOLATE_RELPERM:INTERPOLATE_GO",
                        "RMSGLOBPARAMS:COHIBA_MODEL_MODE",
                        "GENPARAM*",
                        "FIELDPARAM1",
                    ],
                },
            },
            {
                "name": "CORRELATION3",
                "obs_group": {
                    "add": "OP_*_*",
                    "remove": ["OP_*_WGOR*", "SHUT_IN_*", "ROP_*"],
                },
                "model_group": {
                    "add": ["MULTFLT", "MULTZ"],
                    "remove": ["MULTFLT:MULTFLT_F1", "MULTFLT:MULTFLT_F5"],
                },
            },
        ]
    }
    conf = LocalisationConfig(
        observations=ert_list_all_obs, parameters=ert_param_dict, **all_kw
    )

    correlation_specification_reference1 = {
        "CORRELATION1": {
            "obs_list": ["OP_1_WWCT1", "OP_2_WWCT1"],
            "param_list": ["MULTZ:MULTZ_LOWREEK", "MULTZ:MULTZ_MIDREEK"],
        },
        "CORRELATION2": {
            "obs_list": [
                "OP_1_WGORT1",
                "OP_1_WGORT2",
                "OP_1_WGORT3",
                "OP_1_WWCT1",
                "OP_1_WWCT2",
                "OP_1_WWCT3",
                "OP_2_WGORT1",
                "OP_2_WGORT2",
                "OP_2_WGORT3",
                "OP_2_WWCT1",
                "OP_2_WWCT2",
                "OP_2_WWCT3",
            ],
            "param_list": [
                "INTERPOLATE_RELPERM:INTERPOLATE_WO",
                "MULTFLT:MULTFLT_F3",
                "MULTFLT:MULTFLT_F4",
                "MULTFLT:MULTFLT_F5",
                "RMSGLOBPARAMS:FWL",
            ],
        },
        "CORRELATION3": {
            "obs_list": [
                "OP_1_WWCT1",
                "OP_1_WWCT2",
                "OP_1_WWCT3",
                "OP_2_WWCT1",
                "OP_2_WWCT2",
                "OP_2_WWCT3",
            ],
            "param_list": [
                "MULTFLT:MULTFLT_F2",
                "MULTFLT:MULTFLT_F3",
                "MULTFLT:MULTFLT_F4",
                "MULTZ:MULTZ_LOWREEK",
                "MULTZ:MULTZ_MIDREEK",
            ],
        },
    }
    correlation_specification = local.read_correlation_specification(
        all_kw, ert_list_all_obs, ert_param_dict
    )
    local.debug_print(
        f"  -- Test1:  correlation_specification\n  {correlation_specification}\n\n"
    )
    local.debug_print(f"-- correlation_specification: {correlation_specification}\n")
    local.debug_print(
        f"-- correlation_specification_reference: "
        f"{correlation_specification_reference1}\n"
    )
    assert correlation_specification == correlation_specification_reference1

    for corr in conf.correlations:
        correlation_specification_reference1[corr.name][
            "obs_list"
        ] == corr.obs_group.add
        correlation_specification_reference1[corr.name][
            "param_list"
        ] == corr.model_group.add

    # Test 2:
    all_kw = {
        "correlations": [
            {
                "name": "CORRELATION1",
                "obs_group": {"add": ["OP_1_GROUP", "OP_2_GROUP"]},
                "model_group": {
                    "add": ["MULTZ_GROUP", "MULTFLT_GROUP1"],
                    "remove": ["MULTFLT:MULTFLT_F2", "MULTFLT:MULTFLT_F3"],
                },
            },
            {
                "name": "CORRELATION2",
                "obs_group": {
                    "add": ["OP_2_GROUP", "ROP_*_OBS"],
                    "remove": ["OP_2_WGORT2"],
                },
                "model_group": {
                    "add": [
                        "RMSGLOBPARAMS_GROUP",
                        "INTERPOLATE_RELPERM_GROUP",
                        "MULTZ",
                        "MULTFLT",
                        "GENPARAM_GROUP",
                        "FIELDPARAM1",
                    ],
                    "remove": [
                        "MULTFLT:MULTFLT_F1",
                        "MULTFLT:MULTFLT_F2",
                        "INTERPOLATE_RELPERM:INTERPOLATE_GO",
                        "GENPARAM1",
                    ],
                },
            },
        ]
    }
    correlation_specification_reference2 = {
        "CORRELATION1": {
            "obs_list": [
                "OP_1_WWCT1",
                "OP_1_WWCT2",
                "OP_1_WWCT3",
                "OP_2_WWCT1",
                "OP_2_WWCT2",
                "OP_2_WWCT3",
            ],
            "param_list": [
                "MULTFLT:MULTFLT_F1",
                "MULTZ:MULTZ_LOWREEK",
                "MULTZ:MULTZ_MIDREEK",
            ],
        },
        "CORRELATION2": {
            "obs_list": ["OP_2_WWCT1", "OP_2_WWCT2", "OP_2_WWCT3", "ROP_1_OBS"],
            "param_list": [
                "FIELDPARAM1",
                "GENPARAM21",
                "INTERPOLATE_RELPERM:INTERPOLATE_WO",
                "MULTFLT:MULTFLT_F3",
                "MULTFLT:MULTFLT_F4",
                "MULTFLT:MULTFLT_F5",
                "MULTZ:MULTZ_LOWREEK",
                "MULTZ:MULTZ_MIDREEK",
                "RMSGLOBPARAMS:FWL",
            ],
        },
    }
    correlation_specification = local.read_correlation_specification(
        all_kw, ert_list_all_obs, ert_param_dict, obs_group_dict, param_group_dict
    )
    local.debug_print(
        f"  -- Test2:  correlation_specification\n  {correlation_specification}\n\n"
    )
    local.debug_print(f"-- correlation_specification: {correlation_specification}\n")
    local.debug_print(
        f"-- correlation_specification_reference: "
        f"{correlation_specification_reference2}\n"
    )
    assert correlation_specification == correlation_specification_reference2


@pytest.mark.parametrize(
    "param_name, expected",
    [
        ("MULTFLT:MULTFLT_F1", ("MULTFLT", "MULTFLT_F1", 0)),
        ("MULTFLT:MULTFLT_F4", ("MULTFLT", "MULTFLT_F4", 3)),
        ("MULTFLT:MULTFLT_F50", ("MULTFLT", "MULTFLT_F50", 4)),
        ("MULTZ:MULTZ2", ("MULTZ", "MULTZ2", 1)),
        ("MULTZ:MULTZ3", ("MULTZ", "MULTZ3", 2)),
        ("MULTX:MULTX1", ("MULTX", "MULTX1", 0)),
        ("FIELDPARAM1", ("FIELDPARAM1", "FIELDPARAM1", None)),
        ("GENPARAM1", ("GENPARAM1", "GENPARAM1", None)),
    ],
)
def test_active_index_for_parameter(param_name, expected):
    ert_param_dict = {
        "MULTFLT": [
            "MULTFLT_F1",
            "MULTFLT_F2",
            "MULTFLT_F3",
            "MULTFLT_F4",
            "MULTFLT_F50",
        ],
        "MULTZ": ["MULTZ1", "MULTZ2", "MULTZ3"],
        "MULTX": ["MULTX1"],
        "FIELDPARAM1": None,
        "GENPARAM1": None,
    }
    expected_result = expected
    result = local.active_index_for_parameter(param_name, ert_param_dict)
    assert result == expected_result


@pytest.mark.parametrize(
    "obs_list1, param_list1, obs_list2, param_list2, expected",
    [
        (
            ["OBS1", "OBS2", "OBS3"],
            ["PA:PA1", "PA:PA2", "PA:PA3"],
            ["OBS1", "OBS2", "OBS3", "OBS4"],
            ["PB:PB1", "PB:PB2"],
            0,
        ),
        (
            ["OBS1", "OBS2", "OBS3"],
            ["PA:PA1", "PA:PA2", "PA:PA3"],
            ["OBS1", "OBS2", "OBS3", "OBS4"],
            ["PB:PB1", "PA:PA2"],
            3,
        ),
        (
            ["OBS1", "OBS2", "OBS3"],
            ["PA:PA1", "PA:PA2", "PA:PA3"],
            ["OBS1", "OBS2", "OBS3", "OBS4"],
            ["PA:PA1", "PA:PA2"],
            6,
        ),
        (
            ["OBS1", "OBS12", "OBS3"],
            ["PA:PA1", "PA:PA2", "PA:PA3"],
            ["OBS11", "OBS12", "OBS13", "OBS14"],
            ["PA:PA1", "PA:PA2"],
            2,
        ),
        (
            ["OBS1", "OBS2", "OBS3"],
            ["PA:PA1", "PA:PA2", "PA:PA3"],
            ["OBS10", "OBS20", "OBS30", "OBS40"],
            ["PA:PA1", "PA:PA2"],
            0,
        ),
        (
            ["OBS1", "OBS2", "OBS3"],
            ["PA:PA1", "PB:PB2", "PA:PA3"],
            ["OBS1", "OBS12", "OBS13", "OBS14"],
            ["PB:PB1", "PB:PB2"],
            1,
        ),
    ],
)
def test_check_for_duplicated_correlation_specifications(
    obs_list1, param_list1, obs_list2, param_list2, expected
):
    correlation_dict = {
        "CORR1": {
            "obs_list": obs_list1,
            "param_list": param_list1,
        },
        "CORR2": {
            "obs_list": obs_list2,
            "param_list": param_list2,
        },
    }
    expected_result = expected
    result = local.check_for_duplicated_correlation_specifications(correlation_dict)
    assert result == expected_result


@pytest.mark.parametrize(
    "obs_list1, param_list1,  expected",
    [
        (["OBS1", "OBS2", "OBS3"], ["PA:PA1", "PA:PA2", "PA:PA3"], 0),
        (["OBS1", "OBS2", "OBS3"], ["PA:PA1", "PA:PA1", "PA:PA3"], 3),
        (["OBS1", "OBS2", "OBS3"], ["PA:PA1", "PA:PA2", "PA:PA1"], 3),
        (["OBS1", "OBS1", "OBS1"], ["PA:PA1", "PA:PA1", "PA:PA1"], 8),
        (["OBS1", "OBS12", "OBS1"], ["PA:PA1", "PA:PA11", "PA:PA1"], 5),
        (["OBS1"], ["PA:PA1", "PA:PA1", "PA:PA12"], 1),
        (["OBS1"], ["PA:PA1"], 0),
        (["OBS1", "OBS2"], ["PA:PA1"], 0),
    ],
)
def test_check_for_duplicated_correlation_specifications2(
    obs_list1, param_list1, expected
):
    correlation_dict = {
        "CORR1": {
            "obs_list": obs_list1,
            "param_list": param_list1,
        },
    }
    expected_result = expected
    result = local.check_for_duplicated_correlation_specifications(correlation_dict)
    assert result == expected_result


@pytest.mark.parametrize(
    "obs_list1, param_list1, obs_list2, param_list2, obs_list3, param_list3",
    [
        (
            ["OBS1", "OBS2"],
            ["PA:PA1", "PA:PA2", "PA:PA3"],
            ["OBS2", "OBS3", "OBS4"],
            ["PB:PB1", "PB:PB2"],
            ["OBS1", "OBS4"],
            ["PC:PC1", "FIELDPARAM1", "GENPARAM1"],
        ),
    ],
)
def test_add_ministeps(
    obs_list1, param_list1, obs_list2, param_list2, obs_list3, param_list3
):
    ert_mock = Mock()
    local_mock = Mock()
    mock_ministep = Mock()
    mock_update_step = Mock()
    mock_model_param_group = Mock()
    mock_obs_group = Mock()
    mock_active_list = Mock()

    ert_mock.getLocalConfig.return_value = local_mock
    local_mock.createMinistep.return_value = mock_ministep
    local_mock.createDataset.return_value = mock_model_param_group
    local_mock.createObsdata.return_value = mock_obs_group
    local_mock.getUpdatestep.return_value = mock_update_step

    mock_model_param_group.addNode.return_value = None
    mock_model_param_group.getActiveList.return_value = mock_active_list
    mock_obs_group.addNode.return_value = None
    mock_active_list.addActiveIndex.return_value = None

    mock_ministep.attachDataset.return_value = None
    mock_ministep.attachObsset.return_value = None
    mock_update_step.attachMinistep.return_value = None

    ert_param_dict = {
        "PA": [
            "PA1",
            "PA2",
            "PA3",
        ],
        "PB": [
            "PB1",
            "PB2",
        ],
        "PC": [
            "PC1",
        ],
        "FIELDPARAM1": None,
        "GENPARAM1": None,
    }

    correlation_specification = {
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {"add": obs_list1},
                "param_group": {"add": param_list1},
            },
            {
                "name": "CORR2",
                "obs_group": {"add": obs_list2},
                "param_group": {"add": param_list2},
            },
            {
                "name": "CORR3",
                "obs_group": {"add": obs_list3},
                "param_group": {"add": param_list3},
            },
        ]
    }

    class UserConfig(BaseModel):
        correlations: List[CorrelationConfig]

    config = UserConfig(**correlation_specification)
    local.add_ministeps(config, ert_param_dict, ert_mock)
    assert local_mock.createMinistep.called_once()
    assert mock_ministep.attachDataset.called_once_with(mock_model_param_group)
    assert mock_ministep.attachObsset.called_once_with(mock_obs_group)
    assert mock_update_step.attachMinistep.called_once_with(mock_ministep)


def test_clear_correlations():
    ert_mock = Mock()
    local_mock = Mock()
    ert_mock.getLocalConfig.return_value = local_mock
    local_mock.clear.return_value = None
    local.clear_correlations(ert_mock)
    assert local_mock.clear.called_once()
