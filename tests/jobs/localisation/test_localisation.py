from unittest.mock import Mock
import os
import shutil

import yaml
import pytest
from res.enkf import EnKFMain, ResConfig

import semeio.workflows.localisation as localisation
from semeio.workflows.localisation.local_config_scalar import create_ministep
import semeio.workflows.localisation.local_script_lib as local


def test_create_ministep():
    local_mock = Mock()
    mock_ministep = Mock()
    mock_model_group = Mock()
    mock_obs_group = Mock()
    mock_update_step = Mock()

    local_mock.createMinistep.return_value = mock_ministep
    mock_ministep.attachDataset.return_value = None
    mock_ministep.attachObsset.return_value = None
    mock_model_group.name.return_value = "OBS"
    mock_obs_group.name.return_value = "PARA"

    obs = "OBS"
    para = "PARA"
    create_ministep(
        local_config=local_mock,
        mini_step_name="TEST",
        model_group=mock_model_group,
        obs_group=mock_obs_group,
        updatestep=mock_update_step,
    )
    assert local_mock.createMinistep.called_once()
    assert mock_ministep.attachDataset.called_once_with(para)
    assert mock_ministep.attachObsset.called_once_with(obs)
    assert mock_update_step.attachMinistep.called_once_with(mock_ministep)


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
    obs_list = obs_list
    obs_group = {"obs_groups": [{"name": "OBS_GROUP", "add": obs_group_to_add}]}

    with pytest.raises(ValueError) as err_msg:
        local.read_obs_groups(obs_list, obs_group)

    assert expected_error in str(err_msg.value)


def test_read_obs_groups():
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
    local.debug_print(f" obs_groups: {obs_groups}")
    local.debug_print(f" obs_groups_reference: {obs_groups_reference}")

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
    local.debug_print(f" -- obs_list: {obs_list}")
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
    local.debug_print(f" -- obs_list: {obs_list}")
    assert obs_list == obs_list_reference3


@pytest.mark.usefixtures("setup_tmpdir")
def test_localisation_run(test_data_root):
    """test data_set with only scalar parameters"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    res_config.convertToCReference(None)
    ert = EnKFMain(res_config)

    config = {
        "localisation": {
            "config_path": "./",  # Required
            "add_correlations": [
                {
                    "name": "IRRELEVANT",
                    "obs_group": "SOME_NAME",
                    "model_group": "SOME_NAME",
                }
            ],
            "model_param_groups": [  # Required
                {
                    "name": "SOME_NAME",
                    "type": "Scalar",
                    "nodes": [{"name": "SNAKE_OIL_PARAM", "all_active": "YES"}],
                }
            ],
            "obs_groups": [{"name": "SOME_NAME", "nodes": ["WOPR_OP1_9"]}],
        }
    }
    with open("config.yml", "w") as fh:
        yaml.dump(config, fh)

    localisation.LocalisationConfigJob(ert).run("config.yml")
