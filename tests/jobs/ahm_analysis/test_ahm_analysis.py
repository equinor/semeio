import os

import cwrap
import pytest
import pandas as pd
import numpy as np

import semeio.workflows.ahm_analysis.ahmanalysis as ahmanalysis


from scipy import stats
from ecl import EclDataType
from ecl.eclfile import EclKW
from ecl.grid import EclGridGenerator


@pytest.mark.usefixtures("setup_tmpdir")
def test_make_update_log_df(test_data_root):
    """test function creates a dataframe from update_log file
    and replace '...' with key_obs"""
    test_data_dir = os.path.join(test_data_root, "update_log/oneobs")
    update_log_path = test_data_dir
    key_obs = "RWI_3_OBS"
    updatelog_obs = ahmanalysis.make_update_log_df(update_log_path, key_obs)
    assert "..." not in updatelog_obs["obs_key"]
    assert updatelog_obs.at[0, "status"] == "Active"
    assert updatelog_obs.at[1, "status"] == "Inactive"
    assert (
        updatelog_obs.columns
        == [
            "obs_number",
            "obs_key",
            "obs_mean",
            "obs_std",
            "status",
            "sim_mean",
            "sim_std",
        ]
    ).all()


@pytest.mark.parametrize(
    "input_dir, obs_key, expected_result",
    [
        ("update_log/oneobs", "RWI_3_OBS", "2 active/3"),
        ("update_log/allobs", "ALL_OBS", "41 active/47"),
        ("update_log/noactive", "RWI_2_OBS", "0 active/3"),
    ],
)
@pytest.mark.usefixtures("setup_tmpdir")
def test_list_active_observations(test_data_root, input_dir, expected_result, obs_key):
    """test function creates a dataframe reporting active observations during update"""
    test_data_dir = os.path.join(test_data_root, input_dir)
    active_obs = pd.DataFrame()
    df_update_log = ahmanalysis.make_update_log_df(test_data_dir, obs_key)
    active_obs = ahmanalysis.list_active_observations(
        obs_key, active_obs, df_update_log
    )
    assert active_obs.at["ratio", obs_key] == expected_result


@pytest.mark.usefixtures("setup_tmpdir")
def test_create_path(test_data_root):
    """test function creates a path to store data"""
    runpath = os.path.join(test_data_root, "snake_oil")
    runpath2 = os.path.join(test_data_root, "snake_oil/")
    target_name = "testing_case"
    outputpath = ahmanalysis.create_path(runpath, target_name)
    assert "/share/output_analysis/scalar_" in outputpath
    assert target_name in outputpath
    assert outputpath[-1] == "/"
    outputpath2 = ahmanalysis.create_path(runpath2, target_name)
    assert "/share/output_analysis/scalar_" in outputpath2
    assert target_name in outputpath2


@pytest.mark.parametrize(
    "input_obs, expected_result, update_log",
    [
        (["RWI_3_OBS"], [2.0], {"status": {0: "Active", 1: "Inactive", 2: "Active"}}),
        (
            ["RWI_3_OBS", "OP_3_WWCT1"],
            [3.5 / 4.0],
            {"status": {0: "Active", 1: "Inactive", 2: "Active", 3: "Active"}},
        ),
        (["OP_3_WWCT1"], [1.0], {"status": {0: "Active"}}),
    ],
)
def test_list_observations_misfit(input_obs, expected_result, update_log):
    """test function creates a dataframe
    reporting misfit data for each obs vector except the ones with All_obs-"""
    misfit_df = pd.DataFrame(
        {
            "MISFIT:RWI_3_OBS": [6],
            "MISFIT:OP_3_WWCT1": [1],
            "MISFIT:OP_3_WWCT2": [0.15],
            "MISFIT:OP_3_WWCT3": [0.25],
        }
    )
    update_log_df = pd.DataFrame(update_log)
    misfitval = ahmanalysis.list_observations_misfit(
        input_obs, update_log_df, misfit_df
    )
    assert misfitval == expected_result


@pytest.mark.parametrize(
    "input_parameter, expected_result",
    [
        ("PORO", [np.array(8 * [float(nr)]) for nr in range(5, 10)]),
        ("PERMX", [np.array(8 * [float(nr)]) for nr in range(5)]),
    ],
)
@pytest.mark.usefixtures("setup_tmpdir")
def test_import_field_param(input_parameter, expected_result):
    """test function reads field parameter files and creates a dataframe from it"""

    def flatten(regular_list):
        return [item for sublist in regular_list for item in sublist]

    grid = EclGridGenerator.createRectangular((2, 2, 2), (1, 1, 1))
    grid.save_EGRID("MY_GRID.EGRID")

    for iens in range(5):
        permx = EclKW("PERMX", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        permx.assign(iens)

        poro = EclKW("PORO", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        poro.assign(iens + 5)

        with cwrap.open("%d_PERMX_field.grdecl" % iens, "w") as f:
            permx.write_grdecl(f)

        with cwrap.open("%d_PORO_field.grdecl" % iens, "w") as f:
            poro.write_grdecl(f)
    files = [f"{ens_nr}_{input_parameter}_field.grdecl" for ens_nr in range(5)]
    result = ahmanalysis._import_field_param("MY_GRID", input_parameter, files)
    assert flatten(result) == flatten(expected_result)


def test_calc_delta_grid():
    """test function creates a dataframe reporting mean
    delta grids for field parameters"""
    all_input_post = [
        [1 + i for i in range(8)]
        + [10 + i for i in range(8)]
        + [20 + i for i in range(8)],
        [3 + i for i in range(8)]
        + [30 + i for i in range(8)]
        + [40 + i for i in range(8)],
    ]
    all_input_prior = [
        [2 + i for i in range(8)]
        + [20 + i for i in range(8)]
        + [30 + i for i in range(8)],
        [6 + i for i in range(8)]
        + [60 + i for i in range(8)]
        + [80 + i for i in range(8)],
    ]
    mygrid_ok = pd.DataFrame(
        {
            "IX": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
            "JY": [
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            ],
            "KZ": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )
    caseobs = "PORO"
    mygrid_ok_short = pd.DataFrame(
        {
            "IX": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "JY": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "KZ": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    mygrid_ok_short = ahmanalysis.calc_delta_grid(
        all_input_post, all_input_prior, mygrid_ok, caseobs, mygrid_ok_short
    )
    assert mygrid_ok_short["Mean_D_PORO"].max() == 25.0
    assert mygrid_ok_short["Mean_D_PORO"].min() == 2.0
    assert mygrid_ok_short[mygrid_ok_short["IX"] == 1]["Mean_D_PORO"].mean() == 20.0
    assert mygrid_ok_short[mygrid_ok_short["JY"] == 1]["Mean_D_PORO"].mean() == 47 / 3


def test_calc_ks():
    """test function creates a dataframe reporting
    ks value between 0 and 1 for a prior and posterior distribution"""
    ks_matrix = pd.DataFrame()
    dkeys = ["param1", "param2"]
    np.random.seed(12345678)
    prior_data = pd.DataFrame(
        {
            "param1": stats.norm.rvs(size=100, loc=0, scale=1),
            "param2": stats.norm.rvs(size=100, loc=0.01, scale=1),
        }
    )
    target_data = pd.DataFrame(
        {
            "param1": stats.norm.rvs(size=200, loc=0.5, scale=1.5),
            "param2": stats.norm.rvs(size=200, loc=0.1, scale=1),
        }
    )
    key = "WOPT:W1"
    ks_matrix = ahmanalysis.calc_ks(ks_matrix, dkeys, prior_data, target_data, key)
    assert "param2" in ks_matrix.index
    assert ks_matrix.loc["param1", "WOPT:W1"] == 0.18
    assert ks_matrix["WOPT:W1"].max() <= 1
    assert ks_matrix["WOPT:W1"].min() >= 0


def test_check_names():
    """test function check names to be used for prior and update case"""
    ert_currentname = "default"
    prior_name = None
    target_name = "<ANALYSIS_CASE_NAME>"
    p_name, tname = ahmanalysis.check_names(ert_currentname, prior_name, target_name)
    assert p_name == ert_currentname
    assert tname != "<ANALYSIS_CASE_NAME>"


def test_check_inputs():
    """test function check that run fails if empty misfit or prior data"""
    misfitdata_empty = pd.DataFrame()
    misfitdata = pd.DataFrame(
        {
            "MISFIT:RWI_3_OBS": [6],
            "MISFIT:OP_3_WWCT1": [0.05],
            "MISFIT:OP_3_WWCT2": [0.15],
            "MISFIT:OP_3_WWCT3": [0.25],
        }
    )
    prior_data_empty = pd.DataFrame()
    prior_data = pd.DataFrame(
        {
            "SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_OCTAVES": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_OFFSET": [0, 0, 0],
            "SNAKE_OIL_PRES:BPR_138_PERSISTENCE": [0, 1, 2],
        }
    )
    output = ahmanalysis.check_inputs(misfitdata, prior_data)
    # check that it fails if empty
    with pytest.raises(Exception):
        ahmanalysis.check_inputs(misfitdata_empty, prior_data)
    with pytest.raises(Exception):
        ahmanalysis.check_inputs(misfitdata, prior_data_empty)
    assert output.empty is False


def test_initialize_emptydf():
    df1, df2, df3 = ahmanalysis.initialize_emptydf()
    assert df1.empty
    assert df2.empty
    assert df3.empty


@pytest.mark.parametrize(
    "input_map, expected_keys",
    [
        (
            {"data_key_1": ["obs_1", "obs_2", "obs_3"]},
            ["data_key_1"],
        ),
        (
            {"data_key_1": ["obs_1", "obs_2", "obs_3"], "data_key_2": ["obs_4"]},
            ["data_key_1", "data_key_2", "All_obs"],
        ),
        (
            {
                "data_key_1": ["obs_1", "obs_2", "obs_3"],
                "data_key_2": ["obs_4"],
                "data_key_3": ["obs_6"],
            },
            [
                "data_key_1",
                "data_key_2",
                "data_key_3",
                "All_obs-data_key_3",
                "All_obs-data_key_2",
                "All_obs-data_key_1",
                "All_obs",
            ],
        ),
    ],
)
def test_make_obs_groups(input_map, expected_keys):
    result = ahmanalysis.make_obs_groups(input_map)
    assert list(result.keys()) == expected_keys


@pytest.mark.usefixtures("setup_tmpdir")
def test_get_field_grid_char(test_data_root):
    """test function creates a dataframe
    with the grid characteristics and check if fails if wrong format"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    grid_path = test_data_dir + "/grid/SNAKE_OIL_FIELD.EGRID"
    mygrid_ok = ahmanalysis.get_field_grid_char(grid_path)

    assert mygrid_ok["IX"].nunique() == 10
    assert mygrid_ok["JY"].nunique() == 12
    assert mygrid_ok["KZ"].nunique() == 5
    # check that it fails if wrong format
    mygrid_fail = "../../legacy_test_data/snake_oil/grid/SNAKE_OIL_FIELD.ROFF"
    with pytest.raises(Exception):
        ahmanalysis.get_field_grid_char(mygrid_fail)


def test_get_updated_parameters():
    """test function creates a dataframe with all scalar parameters"""

    prior_data = pd.DataFrame(
        {
            "SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_OCTAVES": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_OFFSET": [0, 0, 0],
            "SNAKE_OIL_PRES:BPR_138_PERSISTENCE": [0, 1, 2],
        }
    )
    scalar_parameters = ["SNAKE_OIL_WELL", "SNAKE_OIL_PRES"]
    p_keysf = ahmanalysis.get_updated_parameters(prior_data, scalar_parameters)
    updated_params = [
        "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
        "SNAKE_OIL_PARAM:OP1_OCTAVES",
        "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
        "SNAKE_OIL_PARAM:OP1_OFFSET",
        "SNAKE_OIL_PRES:BPR_138_PERSISTENCE",
    ]
    assert all(item in updated_params for item in p_keysf)
    prior_data2 = pd.DataFrame(
        {
            "SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_OCTAVES": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": [0, 1, 2],
            "SNAKE_OIL_PARAM:OP1_OFFSET": [0, 0, 0],
            "SNAKE_OIL_PRES:BPR_138_PERSISTENCE": [0, 1, 2],
            "LOG10_SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
        }
    )
    p_keysf2 = ahmanalysis.get_updated_parameters(prior_data2, scalar_parameters)
    # check the constant and transformed parameters are removed from list
    assert "SNAKE_OIL_PARAM:OP1_OFFSET" not in p_keysf2
    assert "LOG_SNAKE_OIL_PARAM:OP1_PERSISTENCE" not in p_keysf2
