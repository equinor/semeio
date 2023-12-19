import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from ert import LibresFacade
from ert.storage import open_storage

from semeio._exceptions.exceptions import ValidationError
from semeio.workflows.ahm_analysis import ahmanalysis


def test_ahmanalysis_run(snake_oil_facade):
    """test data_set with only scalar parameters"""
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            ahmanalysis.AhmAnalysisJob,
            storage,
            storage.get_ensemble_by_name("default"),
            prior_name="default",
        )

    # assert that this returns/generates a KS csv file
    output_dir = Path("storage/snake_oil/reports/snake_oil/default/AhmAnalysisJob")
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
        "FIELD_PERMX",
        "FIELD_PORO",
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


def test_ahmanalysis_run_field(snake_oil_facade):
    """test data_set with scalar and Field parameters"""
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        snake_oil_facade.run_ertscript(
            ahmanalysis.AhmAnalysisJob,
            storage,
            storage.get_ensemble_by_name("default"),
            prior_name="default",
        )
    # assert that this returns/generates the delta field parameter
    obs_keys = ["WPR_DIFF_1", "FOPR", "WOPR:OP1"]
    output_deltafield = os.path.join(
        "storage",
        "snake_oil",
        "reports",
        "snake_oil",
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
def test_that_dataset_with_no_prior_will_fail(test_data_root):
    test_data_dir = os.path.join(test_data_root, "snake_oil")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))
    ert = LibresFacade.from_config_file("snake_oil.ert")
    expected_msg = "Empty prior ensemble"
    with pytest.raises(ValidationError, match=expected_msg):
        with open_storage(ert.enspath, "w") as storage:
            ert.run_ertscript(
                ahmanalysis.AhmAnalysisJob,
                storage,
                storage.create_experiment().create_ensemble(
                    name="default", ensemble_size=ert.get_ensemble_size()
                ),
                prior_name="default",
            )
