import os
import shutil
import pytest
import pandas as pd
import semeio.workflows.ahm_analysis.ahmanalysis as ahmanalysis
from semeio._exceptions.exceptions import ValidationError

from pathlib import Path
from res.enkf import EnKFMain, ResConfig
from res.enkf.export import (
    SummaryObservationCollector,
    GenDataObservationCollector,
)
from ecl.grid import EclGridGenerator


@pytest.mark.usefixtures("setup_tmpdir")
def test_ahmanalysis_run(test_data_root):
    """test data_set with only scalar parameters"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    res_config.convertToCReference(None)
    ert = EnKFMain(res_config)

    ahmanalysis.AhmAnalysisJob(ert).run(prior_name="default_0")

    # assert that this returns/generates a KS csv file
    output_dir = Path("storage/snake_oil/reports/default_0/AhmAnalysisJob")
    group_obs = [
        "FOPR",
        "WOPR_OP1",
        "SNAKE_OIL_WPR_DIFF",
        "All_obs",
        "All_obs-SNAKE_OIL_WPR_DIFF",
        "All_obs-WOPR_OP1",
        "All_obs-FOPR",
    ]
    parameters = [
        "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
        "SNAKE_OIL_PARAM:OP1_OCTAVES",
        "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
        "SNAKE_OIL_PARAM:OP1_OFFSET",
        "SNAKE_OIL_PARAM:OP2_PERSISTENCE",
        "SNAKE_OIL_PARAM:OP2_OCTAVES",
        "SNAKE_OIL_PARAM:OP2_DIVERGENCE_SCALE",
        "SNAKE_OIL_PARAM:OP2_OFFSET",
        "SNAKE_OIL_PARAM:BPR_555_PERSISTENCE",
        "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
    ]
    assert (output_dir / "ks.csv").is_file()
    ks_df = pd.read_csv(output_dir / "ks.csv")
    for keys in ks_df["Parameters"].tolist():
        assert keys in parameters
    assert ks_df.columns[1:].tolist() == group_obs
    assert ks_df["WOPR_OP1"].max() <= 1
    assert ks_df["WOPR_OP1"].min() >= 0
    assert (output_dir / "active_obs_info.csv").is_file()
    assert (output_dir / "misfit_obs_info.csv").is_file()
    assert (output_dir / "prior.csv").is_file()
    for group in group_obs:
        filename = group + ".csv"
        assert (output_dir / filename).is_file()


@pytest.mark.usefixtures("setup_tmpdir")
def test_ahmanalysis_run_field(test_data_root, grid_prop):
    """test data_set with scalar and Field parameters"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))
    os.makedirs("fields")
    grid = EclGridGenerator.createRectangular((10, 12, 5), (1, 1, 1))
    for iens in range(10):
        grid_prop("PERMX", 10, grid.getGlobalSize(), "fields/permx%d.grdecl" % iens)
        grid_prop("PORO", 0.2, grid.getGlobalSize(), "fields/poro%d.grdecl" % iens)

    res_config = ResConfig("snake_oil_field.ert")
    res_config.convertToCReference(None)
    ert = EnKFMain(res_config)

    ahmanalysis.AhmAnalysisJob(ert).run(prior_name="default")

    # assert that this returns/generates the delta field parameter
    gen_obs_list = GenDataObservationCollector.getAllObservationKeys(ert)
    summary_obs_list = SummaryObservationCollector.getAllObservationKeys(ert)
    obs_keys = gen_obs_list + summary_obs_list
    output_deltafield = os.path.join(
        "storage",
        "snake_oil_field",
        "reports",
        "default",
        "AhmAnalysisJob",
        "delta_fieldPERMX.csv",
    )
    assert os.path.isfile(output_deltafield)
    delta_df = pd.read_csv(output_deltafield, index_col=0)
    assert len(delta_df.columns) == 8 + (len(obs_keys) * 2) + 1
    # check field parameter is present and not empty in the final KS matrix
    output_ks = output_deltafield.replace("delta_fieldPERMX.csv", "ks.csv")
    ks_df = pd.read_csv(output_ks, index_col=0)
    assert not ks_df.empty
    assert "FIELD_PERMX" in ks_df.index.tolist()
    check_empty = ks_df.loc[["FIELD_PERMX"], :].isnull().all(axis=1)
    assert not check_empty["FIELD_PERMX"]


@pytest.mark.usefixtures("setup_tmpdir")
def test_no_prior(test_data_root):
    """check dataset without prior data"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))
    res_config = ResConfig("snake_oil.ert")
    res_config.convertToCReference(None)
    ert = EnKFMain(res_config)
    expected_msg = "Empty prior ensemble"
    # check that it fails
    with pytest.raises(ValidationError, match=expected_msg):
        ahmanalysis.AhmAnalysisJob(ert).run(prior_name="default")
