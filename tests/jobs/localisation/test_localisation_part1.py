# import os
# import shutil

# import yaml
import pytest

# from res.enkf import EnKFMain, ResConfig

# import semeio.workflows.localisation as localisation
import semeio.workflows.localisation.local_script_lib as local


@pytest.mark.parametrize(
    "obs_group_to_add,expected",
    [
        (["OBS1", "OBS2"], ["OBS1", "OBS2"]),
        (["OBS1"], ["OBS1"]),
        (["OBS*"], ["OBS1", "OBS2"]),
        ([], []),
        ("All", ["OBS1", "OBS2"]),
    ],
)
def test_read_obs_add_group(obs_group_to_add, expected):
    obs_list = ["OBS1", "OBS2"]
    obs_group = {"obs_groups": [{"name": "OBS_GROUP", "add": obs_group_to_add}]}

    expected_result = {"OBS_GROUP": expected}

    result = local.read_obs_groups(obs_list, obs_group)
    assert result == expected_result


@pytest.mark.parametrize(
    "obs_group_to_add, obs_group_to_remove, expected",
    [
        (["OBS1", "OBS2"], ["OBS1"], ["OBS2"]),
        ("All", ["OBS2"], ["OBS1"]),
        ("All", ["OBS*"], []),
    ],
)
def test_read_obs_add_remove_group(obs_group_to_add, obs_group_to_remove, expected):
    obs_list = ["OBS1", "OBS2"]
    obs_group = {
        "obs_groups": [
            {
                "name": "OBS_GROUP",
                "add": obs_group_to_add,
                "remove": obs_group_to_remove,
            }
        ]
    }

    expected_result = {"OBS_GROUP": expected}

    result = local.read_obs_groups(obs_list, obs_group)
    assert result == expected_result


@pytest.mark.parametrize(
    "obs_list, obs_group_to_add, expected_error",
    [
        (["OBS1", "OBS2"], ["Q*1"], "The user defined observations Q*1"),
        ([], [], "all_observations has 0 observations"),
    ],
)
def test_invalid_read_obs(obs_list, obs_group_to_add, expected_error):
    obs_group = {"obs_groups": [{"name": "OBS_GROUP", "add": obs_group_to_add}]}

    with pytest.raises(ValueError) as err_msg:
        local.read_obs_groups(obs_list, obs_group)

    assert expected_error in str(err_msg.value)


def test_read_obs_groups():
    local.debug_print("\n\nRun: test_read_obs_groups")
    ert_list_all_obs = [
        "OP_1_WWCT1",
        "OP_1_WWCT2",
        "OP_1_WWCT3",
        "OP_2_WWCT1",
        "OP_2_WWCT2",
        "OP_2_WWCT3",
        "OP_3_WWCT1",
        "OP_3_WWCT2",
        "OP_3_WWCT3",
        "OP_4_WWCT1",
        "OP_4_WWCT2",
        "OP_4_WWCT3",
        "OP_5_WWCT1",
        "OP_5_WWCT2",
        "OP_5_WWCT3",
        "OP_1_WGORT1",
        "OP_1_WGORT2",
        "OP_1_WGORT3",
        "OP_2_WGORT1",
        "OP_2_WGORT2",
        "OP_2_WGORT3",
        "OP_3_WGORT1",
        "OP_3_WGORT2",
        "OP_3_WGORT3",
        "OP_4_WGORT1",
        "OP_4_WGORT2",
        "OP_4_WGORT3",
        "OP_5_WGORT1",
        "OP_5_WGORT2",
        "OP_5_WGORT3",
        "ROP_1_OBS",
        "ROP_2_OBS",
        "ROP_3_OBS",
        "ROP_4_OBS",
        "ROP_5_OBS",
        "SHUT_IN_OP1",
        "SHUT_IN_OP2",
        "SHUT_IN_OP3",
        "SHUT_IN_OP4",
        "SHUT_IN_OP5",
    ]
    # Lists of observation names are sorted alphabetically
    obs_groups_reference = {
        "OP_1_GROUP": ["OP_1_WWCT1", "OP_1_WWCT2", "OP_1_WWCT3"],
        "OP_2_GROUP": [
            "OP_1_WWCT1",
            "OP_1_WWCT2",
            "OP_1_WWCT3",
            "OP_3_WWCT1",
            "OP_3_WWCT2",
            "OP_3_WWCT3",
            "OP_4_WWCT1",
            "OP_4_WWCT2",
            "OP_4_WWCT3",
            "OP_5_WWCT1",
            "OP_5_WWCT2",
            "OP_5_WWCT3",
        ],
        "OP_ALL_EXCEPT_1_3": [
            "OP_2_WGORT1",
            "OP_2_WGORT2",
            "OP_2_WGORT3",
            "OP_2_WWCT1",
            "OP_2_WWCT2",
            "OP_2_WWCT3",
            "OP_4_WGORT1",
            "OP_4_WGORT2",
            "OP_4_WGORT3",
            "OP_4_WWCT1",
            "OP_4_WWCT2",
            "OP_4_WWCT3",
            "OP_5_WGORT1",
            "OP_5_WGORT2",
            "OP_5_WGORT3",
            "OP_5_WWCT1",
            "OP_5_WWCT2",
            "OP_5_WWCT3",
            "ROP_1_OBS",
            "ROP_2_OBS",
            "ROP_3_OBS",
            "ROP_4_OBS",
            "ROP_5_OBS",
            "SHUT_IN_OP1",
            "SHUT_IN_OP2",
            "SHUT_IN_OP3",
            "SHUT_IN_OP4",
            "SHUT_IN_OP5",
        ],
    }
    all_kw = {}
    obs_group_item1 = {
        "name": "OP_1_GROUP",
        "add": ["OP_1_WWCT1", "OP_1_WWCT2", "OP_1_WWCT3"],
    }
    obs_group_item2 = {
        "name": "OP_2_GROUP",
        "add": ["OP_*_W*"],
        "remove": ["OP_*_WGOR*", "OP_2_W*"],
    }
    obs_group_item3 = {
        "name": "OP_ALL_EXCEPT_1_3",
        "add": "All",
        "remove": ["OP_1_*", "OP_3_*"],
    }

    all_kw["obs_groups"] = [obs_group_item1, obs_group_item2, obs_group_item3]

    obs_groups = local.read_obs_groups(ert_list_all_obs, all_kw)
    local.debug_print(f"  -- Test 1:  obs_groups: {obs_groups}\n\n")
    assert obs_groups == obs_groups_reference

    # Check that Value error is raised
    obs_group_item4 = {
        "name": "OP_FAILURE",
        "add": "All",
        "remove": ["OP_1_*", "O*Q_3_*"],
    }
    all_kw["obs_groups"] = [obs_group_item4]
    try:
        obs_groups = local.read_obs_groups(ert_list_all_obs, all_kw)
        assert False
    except ValueError:
        assert True

    # Check that Value error is raised
    all_kw["obs_groups"] = [obs_group_item3]
    try:
        obs_groups = local.read_obs_groups([], all_kw)
        assert False
    except ValueError:
        assert True


def test_read_obs_groups_for_correlations():
    local.debug_print("\n\nRun: test_read_obs_groups_for_correlations")
    obs_group_dict = {
        "OP_1_GROUP": ["OP_1_WWCT1", "OP_1_WWCT2", "OP_1_WWCT3"],
        "OP_2_GROUP": ["OP_2_WWCT1", "OP_2_WWCT2", "OP_2_WWCT3"],
        "OP_3_GROUP": [
            "OP_3_WGORT1",
            "OP_3_WGORT2",
            "OP_3_WGORT3",
            "OP_3_WWCT1",
            "OP_3_WWCT2",
            "OP_3_WWCT3",
        ],
        "OP_ALL_EXCEPT_1_3": [
            "OP_2_WGORT1",
            "OP_2_WGORT2",
            "OP_2_WGORT3",
            "OP_2_WWCT1",
            "OP_2_WWCT2",
            "OP_2_WWCT3",
            "OP_4_WGORT1",
            "OP_4_WGORT2",
            "OP_4_WGORT3",
            "OP_4_WWCT1",
            "OP_4_WWCT2",
            "OP_4_WWCT3",
            "OP_5_WGORT1",
            "OP_5_WGORT2",
            "OP_5_WGORT3",
            "OP_5_WWCT1",
            "OP_5_WWCT2",
            "OP_5_WWCT3",
            "ROP_1_OBS",
            "ROP_2_OBS",
            "ROP_3_OBS",
            "ROP_4_OBS",
            "ROP_5_OBS",
            "SHUT_IN_OP1",
            "SHUT_IN_OP2",
            "SHUT_IN_OP3",
            "SHUT_IN_OP4",
            "SHUT_IN_OP5",
        ],
    }

    main_keyword = "correlations"
    ert_list_all_obs = [
        "OP_1_WWCT1",
        "OP_1_WWCT2",
        "OP_1_WWCT3",
        "OP_2_WWCT1",
        "OP_2_WWCT2",
        "OP_2_WWCT3",
        "OP_3_WWCT1",
        "OP_3_WWCT2",
        "OP_3_WWCT3",
        "OP_4_WWCT1",
        "OP_4_WWCT2",
        "OP_4_WWCT3",
        "OP_5_WWCT1",
        "OP_5_WWCT2",
        "OP_5_WWCT3",
        "OP_1_WGORT1",
        "OP_1_WGORT2",
        "OP_1_WGORT3",
        "OP_2_WGORT1",
        "OP_2_WGORT2",
        "OP_2_WGORT3",
        "OP_3_WGORT1",
        "OP_3_WGORT2",
        "OP_3_WGORT3",
        "OP_4_WGORT1",
        "OP_4_WGORT2",
        "OP_4_WGORT3",
        "OP_5_WGORT1",
        "OP_5_WGORT2",
        "OP_5_WGORT3",
        "ROP_1_OBS",
        "ROP_2_OBS",
        "ROP_3_OBS",
        "ROP_4_OBS",
        "ROP_5_OBS",
        "SHUT_IN_OP1",
        "SHUT_IN_OP2",
        "SHUT_IN_OP3",
        "SHUT_IN_OP4",
        "SHUT_IN_OP5",
    ]

    # Test 1
    correlation_spec_item = {
        "name": "CORRELATION1",
        "obs_group": {"add": ["OP_1_GROUP", "OP_2_GROUP"]},
    }

    obs_list_reference1 = [
        "OP_1_WWCT1",
        "OP_1_WWCT2",
        "OP_1_WWCT3",
        "OP_2_WWCT1",
        "OP_2_WWCT2",
        "OP_2_WWCT3",
    ]

    obs_list = local.read_obs_groups_for_correlations(
        obs_group_dict, correlation_spec_item, main_keyword, ert_list_all_obs
    )
    local.debug_print(f" -- Test 1: obs_list {obs_list}\n\n")
    assert obs_list == obs_list_reference1

    # Test 2
    correlation_spec_item = {
        "name": "CORRELATION2",
        "obs_group": {"add": ["OP_*_GROUP", "OP_2_GROUP"], "remove": "OP_1_WWCT2"},
    }

    obs_list_reference2 = [
        "OP_1_WWCT1",
        "OP_1_WWCT3",
        "OP_2_WWCT1",
        "OP_2_WWCT2",
        "OP_2_WWCT3",
        "OP_3_WGORT1",
        "OP_3_WGORT2",
        "OP_3_WGORT3",
        "OP_3_WWCT1",
        "OP_3_WWCT2",
        "OP_3_WWCT3",
    ]

    obs_list = local.read_obs_groups_for_correlations(
        obs_group_dict, correlation_spec_item, main_keyword, ert_list_all_obs
    )
    local.debug_print(f" -- Test 2: obs_list {obs_list}\n\n")
    assert obs_list == obs_list_reference2

    # Test 3
    correlation_spec_item = {
        "name": "CORRELATION3",
        "obs_group": {
            "add": ["OP_*_GROUP", "OP_2_GROUP", "ROP_*_OBS"],
            "remove": ["OP_3_WGORT*", "OP_2_WWCT3", "OP_1_GROUP", "ROP_3_OBS"],
        },
    }

    obs_list_reference3 = [
        "OP_2_WWCT1",
        "OP_2_WWCT2",
        "OP_3_WWCT1",
        "OP_3_WWCT2",
        "OP_3_WWCT3",
        "ROP_1_OBS",
        "ROP_2_OBS",
        "ROP_4_OBS",
        "ROP_5_OBS",
    ]

    obs_list = local.read_obs_groups_for_correlations(
        obs_group_dict, correlation_spec_item, main_keyword, ert_list_all_obs
    )
    local.debug_print(f" -- Test 3: obs_list {obs_list}\n\n")
    assert obs_list == obs_list_reference3


@pytest.mark.parametrize(
    "param_to_add, param_to_remove, expected",
    [
        (
            ["NODE*:PA*", "NODE3:PB2"],
            ["NODE*:PB*", "NODE*:PA1"],
            ["NODE1:PA2", "NODE2:PA3", "NODE2:PA4"],
        ),
        (
            "All",
            [],
            [
                "NODE1:PA1",
                "NODE1:PA2",
                "NODE2:PA3",
                "NODE2:PA4",
                "NODE3:PB1",
                "NODE3:PB2",
            ],
        ),
        ("All", ["NODE*"], []),
    ],
)
def test_read_param_add_remove_group(param_to_add, param_to_remove, expected):
    param_dict = {
        "NODE1": ["PA1", "PA2"],
        "NODE2": ["PA3", "PA4"],
        "NODE3": ["PB1", "PB2"],
    }
    param_group = {
        "model_param_groups": [
            {
                "name": "PARAM_GROUP",
                "add": param_to_add,
                "remove": param_to_remove,
            }
        ]
    }

    expected_result = {"PARAM_GROUP": expected}

    result = local.read_param_groups(param_dict, param_group)
    assert result == expected_result


def test_read_param_groups():
    local.debug_print("\n\nRun: test_read_param_groups")
    ert_param_dict = {
        "INTERPOLATE_RELPERM": [
            "INTERPOLATE_WO",
            "INTERPOLATE_GO",
        ],
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
        "GENPARAM2": None,
        "GENPARAM3": None,
    }
    # Test 1:
    all_kw = {}
    param_group_item1 = {"name": "PARAM_GROUP_MULTZ", "add": ["MULTZ:MULTZ_MIDREEK"]}
    param_group_item2 = {
        "name": "PARAM_GROUP_MULTFLT",
        "add": "MULTFLT",
        "remove": ["MULTFLT:MULTFLT_F2", "MULTFLT:MULTFLT_F4"],
    }
    param_group_item3 = {
        "name": "PARAM_GROUP_INTERPOLATE_RELPERM",
        "add": "INTERPOLATE_RELPERM:INTERPOLATE_WO",
    }
    param_group_item4 = {
        "name": "PARAM_GROUP_RMSGLOBPARAMS",
        "add": "RMSGLOBPARAMS",
        "remove": "RMSGLOBPARAMS:COHIBA_MODEL_MODE",
    }
    param_group_item5 = {
        "name": "PARAM_GROUP_GENPARAM",
        "add": ["GENPARAM1", "GENPARAM2", "GENPARAM3"],
        "remove": "GENPARAM2",
    }
    all_kw["model_param_groups"] = [
        param_group_item1,
        param_group_item2,
        param_group_item3,
        param_group_item4,
        param_group_item5,
    ]

    param_groups_reference1 = {
        "PARAM_GROUP_MULTZ": ["MULTZ:MULTZ_MIDREEK"],
        "PARAM_GROUP_MULTFLT": [
            "MULTFLT:MULTFLT_F1",
            "MULTFLT:MULTFLT_F3",
            "MULTFLT:MULTFLT_F5",
        ],
        "PARAM_GROUP_INTERPOLATE_RELPERM": ["INTERPOLATE_RELPERM:INTERPOLATE_WO"],
        "PARAM_GROUP_RMSGLOBPARAMS": ["RMSGLOBPARAMS:FWL"],
        "PARAM_GROUP_GENPARAM": ["GENPARAM1", "GENPARAM3"],
    }
    param_groups = local.read_param_groups(ert_param_dict, all_kw)
    local.debug_print(f"-- Test1: Param groups: {param_groups}\n\n")
    assert param_groups == param_groups_reference1

    # Test 2:
    all_kw = {}
    param_group_item = {
        "name": "PARAM_GROUP_MULTZ2",
        "add": ["MULT*"],
        "remove": ["MULTFLT:MULTFLT_F*"],
    }
    all_kw["model_param_groups"] = [param_group_item]
    param_groups_reference2 = {
        "PARAM_GROUP_MULTZ2": ["MULTZ:MULTZ_LOWREEK", "MULTZ:MULTZ_MIDREEK"]
    }

    param_groups = local.read_param_groups(ert_param_dict, all_kw)
    local.debug_print(f"-- Test2: Param groups: {param_groups}\n\n")
    assert param_groups == param_groups_reference2

    # Test 3:
    all_kw = {}
    param_group_item = {
        "name": "PARAM_GROUP_MULTZ_GENPARAM",
        "add": "All",
        "remove": [
            "MULT*:MULTZ_*",
            "MULTF*_F3",
            "RMSGLOBPARAMS:*",
            "INT*",
            "GENPARAM1",
        ],
    }
    all_kw["model_param_groups"] = [param_group_item]

    param_groups_reference3 = {
        "PARAM_GROUP_MULTZ_GENPARAM": [
            "GENPARAM2",
            "GENPARAM3",
            "MULTFLT:MULTFLT_F1",
            "MULTFLT:MULTFLT_F2",
            "MULTFLT:MULTFLT_F4",
            "MULTFLT:MULTFLT_F5",
        ]
    }

    param_groups = local.read_param_groups(ert_param_dict, all_kw)
    local.debug_print(f"-- Test3: Param groups: {param_groups}\n\n")
    assert param_groups == param_groups_reference3


def test_read_param_groups_for_correlations():
    local.debug_print("\n\nRun: test_read_param_groups_for_correlations")
    main_keyword = "correlations"
    ert_param_dict = {
        "INTERPOLATE_RELPERM": [
            "INTERPOLATE_WO",
            "INTERPOLATE_GO",
        ],
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
        "GENPARAM12": None,
        "GENPARAM22": None,
        "GENPARAM21": None,
    }

    # Test 1
    correlation_spec_item = {
        "name": "CORRELATION1",
        "obs_group": {"add": ["OP_1_GROUP", "OP_2_GROUP"]},
        "model_group": {"add": ["MULTZ_GROUP"], "remove": ["MULTZ:MULTZ_MIDREEK"]},
    }
    param_group_dict = {"MULTZ_GROUP": ["MULTZ:MULTZ_LOWREEK", "MULTZ:MULTZ_MIDREEK"]}

    param_list_reference1 = ["MULTZ:MULTZ_LOWREEK"]

    param_list = local.read_param_groups_for_correlations(
        param_group_dict, correlation_spec_item, main_keyword, ert_param_dict
    )
    local.debug_print(f" -- Test 1: Param list: {param_list}\n\n")
    assert param_list == param_list_reference1

    # Test 2
    correlation_spec_item = {
        "name": "CORRELATION2",
        "obs_group": {"add": ["OP_*_GROUP", "OP_2_GROUP"], "remove": "OP_1_WWCT2"},
        "model_group": {"add": "MULTZ"},
    }
    param_group_dict = {"MULTFLT_GROUP": ["MULTFLT"]}

    param_list_reference2 = ["MULTZ:MULTZ_LOWREEK", "MULTZ:MULTZ_MIDREEK"]

    param_list = local.read_param_groups_for_correlations(
        param_group_dict, correlation_spec_item, main_keyword, ert_param_dict
    )
    local.debug_print(f" -- Test 2: Param list: {param_list}\n\n")
    assert param_list == param_list_reference2

    # Test 3
    correlation_spec_item = {
        "name": "CORRELATION3",
        "obs_group": {
            "add": ["OP_*_GROUP", "OP_2_GROUP", "ROP_*_OBS"],
            "remove": ["OP_3_WGORT*", "OP_2_WWCT3", "OP_1_GROUP", "ROP_3_OBS"],
        },
        "model_group": {
            "add": "All",
            "remove": [
                "R*",
                "I*",
                "M*Z",
                "MULTFLT:M*_F1",
                "MULTFLT:M*_F4",
                "MULTFLT:M*_F5",
                "GENPARAM*2",
            ],
        },
    }
    param_group_dict = {"MULTFLT_GROUP": ["MULTFLT"]}

    param_list_reference3 = [
        "GENPARAM1",
        "GENPARAM21",
        "MULTFLT:MULTFLT_F2",
        "MULTFLT:MULTFLT_F3",
    ]

    param_list = local.read_param_groups_for_correlations(
        param_group_dict, correlation_spec_item, main_keyword, ert_param_dict
    )
    local.debug_print(f" -- Test 3: param_list: {param_list}\n\n")
    assert param_list == param_list_reference3


@pytest.mark.parametrize(
    "user_obs, all_obs, expected",
    [
        (["OBS1"], ["OBS1", "OBS2"], ["OBS1"]),
        (["OBS*"], ["OBS1", "OBS2"], ["OBS1", "OBS2"]),
        (["OBS*"], ["OBS1", "OBS2", "SOME"], ["OBS1", "OBS2"]),
        (["UNKNOWN"], ["OBS1", "OBS2"], []),
    ],
)
def test_expand_wildcards_for_obs(user_obs, all_obs, expected):
    result = local.expand_wildcards(user_obs, all_obs)
    assert result == expected
