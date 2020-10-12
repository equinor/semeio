import os
import shutil
import statistics

import cwrap
import pytest
import pandas as pd
import numpy as np

import semeio.workflows.ahm_analysis.ahmanalysis as ahmanalysis

from res.enkf import (
    EnKFMain,
    ResConfig,
)

from scipy import stats
from ecl import EclDataType
from ecl.eclfile import EclKW
from ecl.grid import EclGridGenerator
from ecl.util.util import RandomNumberGenerator


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


@pytest.mark.usefixtures("setup_tmpdir")
def test_list_active_observations(test_data_root):
    """test function creates a dataframe reporting active observations during update"""
    test_data_dir = os.path.join(test_data_root, "update_log/oneobs")
    test_data_dirsmry = os.path.join(test_data_root, "update_log/allobs")
    test_data_noactiv = os.path.join(test_data_root, "update_log/noactive")
    output_path = test_data_dir.replace("update_log", "scalar_")
    output_pathsmry = test_data_dirsmry.replace("update_log", "scalar_")
    output_pathnoactiv = test_data_noactiv.replace("update_log", "scalar_")
    key_obs = "RWI_3_OBS"
    key_obssmry = "ALL_OBS"
    key_noactiv = "RWI_2_OBS"
    active_obs = pd.DataFrame()
    active_obs = ahmanalysis.list_active_observations(key_obs, active_obs, output_path)
    assert active_obs.at["ratio", "RWI_3_OBS"] == "2 active/3"
    active_obs = ahmanalysis.list_active_observations(
        key_obssmry, active_obs, output_pathsmry
    )
    assert active_obs.at["ratio", "ALL_OBS"] == "41 active/47"
    active_obs = ahmanalysis.list_active_observations(
        key_noactiv, active_obs, output_pathnoactiv
    )
    assert active_obs.at["ratio", "RWI_2_OBS"] == "0 active/3"
    assert (active_obs.columns == ["RWI_3_OBS", "ALL_OBS", "RWI_2_OBS"]).all()


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


@pytest.mark.usefixtures("setup_tmpdir")
def test_list_observations_misfit(test_data_root):
    """test function creates a dataframe
    reporting misfit data for each obs vector except the ones with All_obs-"""
    misfit_df = pd.DataFrame(
        {
            "MISFIT:RWI_3_OBS": [6],
            "MISFIT:OP_3_WWCT1": [0.05],
            "MISFIT:OP_3_WWCT2": [0.15],
            "MISFIT:OP_3_WWCT3": [0.25],
        }
    )
    test_data_dir = os.path.join(test_data_root, "update_log/oneobs")
    test_data_dirsmry = os.path.join(test_data_root, "update_log/allobs")
    test_data_noactiv = os.path.join(test_data_root, "update_log/noactive")
    test_data_activsmry = os.path.join(test_data_root, "update_log/smry")
    output_path = test_data_dir.replace("update_log", "scalar_")
    output_path_allsmry = test_data_dirsmry.replace("update_log", "scalar_")
    output_path_noactiv = test_data_noactiv.replace("update_log", "scalar_")
    output_path_activsmry = test_data_activsmry.replace("update_log", "scalar_")
    gen_obs_list = ["RWI_3_OBS", "RWI_2_OBS"]
    key_obs = ["RWI_3_OBS", "All_obs-RWI_3_OBS", "OP_3_WWCT", "RWI_2_OBS"]
    misfitval = pd.DataFrame()
    misfitval = ahmanalysis.list_observations_misfit(
        key_obs[0], misfitval, gen_obs_list, output_path, misfit_df
    )
    assert misfitval.at["misfit", "RWI_3_OBS"] == 2.0
    misfitval = ahmanalysis.list_observations_misfit(
        key_obs[1], misfitval, gen_obs_list, output_path_allsmry, misfit_df
    )
    assert misfitval.at["misfit", "All_obs-RWI_3_OBS"] == "None"
    misfitval = ahmanalysis.list_observations_misfit(
        key_obs[2], misfitval, gen_obs_list, output_path_activsmry, misfit_df
    )
    assert misfitval.at["misfit", "OP_3_WWCT"] == statistics.mean([0.15, 0.05])
    misfitval = ahmanalysis.list_observations_misfit(
        key_obs[3], misfitval, gen_obs_list, output_path_noactiv, misfit_df
    )
    assert misfitval.at["misfit", "RWI_2_OBS"] == "None"


@pytest.mark.usefixtures("setup_tmpdir")
def test_get_input_state_df(test_data_root):
    """test function reads field parameter files and creates a dataframe from it"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))
    grid = EclGridGenerator.createRectangular((10, 12, 5), (1, 1, 1))
    rng = RandomNumberGenerator()
    rng.setState("ABCD6375ejascEFGHIJ")
    for iens in range(5):
        permx = EclKW("PERMX", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        permx.assign(rng.getDouble())

        poro = EclKW("PORO", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        poro.assign(rng.getDouble())

        if not os.path.isdir("field_out"):
            os.makedirs("field_out")

        with cwrap.open("field_out/%d_PERMX_field.grdecl" % iens, "w") as f:
            permx.write_grdecl(f)

        with cwrap.open("field_out/%d_PORO_field.grdecl" % iens, "w") as f:
            poro.write_grdecl(f)

    data_dir = "field_out/"
    ext = "grdecl"
    #    ext2 = "roff"
    input_grid = os.path.join(test_data_root, "snake_oil/grid/SNAKE_OIL_FIELD.EGRID")
    all_input = ahmanalysis.get_input_state_df(ext, input_grid, 5, data_dir, "PERMX")
    assert len(all_input[0]) == 10 * 12 * 5
    assert len(all_input[4]) == 10 * 12 * 5
    assert len(all_input) == 5
    all_input2 = ahmanalysis.get_input_state_df(ext, input_grid, 5, data_dir, "PORO")
    assert len(all_input2) == 5
    assert len(all_input2[0]) == 10 * 12 * 5
    assert len(all_input2[4]) == 10 * 12 * 5


def test_calc_delta_grid():
    """test function creates a dataframe reporting mean delta grids for field parameters"""
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


# def test_save_to_csv():


@pytest.mark.usefixtures("setup_tmpdir")
def test_make_obs_groups(test_data_root):
    """test function creates a list of observation vectors"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    res_config.convertToCReference(None)
    ert = EnKFMain(res_config)

    obs_keys = ["FOPR", "WOPR:OP1", "WPR_DIFF_1"]
    obs_key_fail = ["All_obs"]
    dfobs = ahmanalysis.make_obs_groups(obs_keys, ert.getObservations())
    assert len(dfobs) == (len(obs_keys) * 2) + 1
    assert len(dfobs["All_obs"]) == 8
    assert len(dfobs["All_obs-WOPR_OP1"]) == 2
    assert all(item in dfobs.keys() for item in ["FOPR", "WOPR_OP1", "WPR_DIFF_1"])
    # check that it  fails if one observation is named "All_obs"
    with pytest.raises(Exception):
        ahmanalysis.make_obs_groups(obs_key_fail, ert.getObservations())


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
            "SNAKE_OIL_PARAM:OP1_OFFSET": [0, 0, 0],
        }
    )
    p_keysf2 = ahmanalysis.get_updated_parameters(prior_data2, scalar_parameters)
    # check the constant and transformed parameters are removed from list
    assert "SNAKE_OIL_PARAM:OP1_OFFSET" not in p_keysf2
    assert "LOG_SNAKE_OIL_PARAM:OP1_PERSISTENCE" not in p_keysf2
