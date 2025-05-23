import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from ert import LibresFacade
from ert.storage import open_storage

from semeio.workflows.ahm_analysis import ahmanalysis


@pytest.mark.integration_test
def test_ahmanalysis_run(snake_oil_facade):
    """test data_set with only scalar parameters"""
    with open_storage("storage/snake_oil/ensemble", "w") as storage:
        ensemble = storage.get_experiment_by_name(
            "ensemble-experiment"
        ).get_ensemble_by_name("default")

        ahmanalysis.AhmAnalysisJob().initializeAndRun(
            argument_types=[],
            argument_values=[],
            fixtures={
                "storage": storage,
                "ensemble": ensemble,
                "reports_dir": (
                    snake_oil_facade.config.analysis_config.log_path
                    / ensemble.experiment.name
                ),
                "observation_settings": snake_oil_facade.config.analysis_config.observation_settings,
                "es_settings": snake_oil_facade.config.analysis_config.es_settings,
                "random_seed": snake_oil_facade.config.random_seed,
            },
        )
    assert not Path("tmp_storage").exists()

    # assert that this returns/generates a KS csv file
    output_dir = Path("log/update/ensemble-experiment/AhmAnalysisJob")
    group_obs = [
        "FOPR",
        "SNAKE_OIL_WPR_DIFF",
        "WOPR_OP1",
        "All_obs",
        "All_obs-WOPR_OP1",
        "All_obs-SNAKE_OIL_WPR_DIFF",
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
    assert sorted(ks_df.columns[1:].tolist()) == sorted(group_obs)
    assert ks_df["WOPR_OP1"].max() <= 1
    assert ks_df["WOPR_OP1"].min() >= 0
    assert (output_dir / "active_obs_info.csv").is_file()
    assert (output_dir / "misfit_obs_info.csv").is_file()
    assert (output_dir / "prior.csv").is_file()
    for group in group_obs:
        filename = group + ".csv"
        assert (output_dir / filename).is_file()


@pytest.mark.integration_test
def test_ahmanalysis_run_group_by_obs(snake_oil_facade):
    """test data_set with only scalar parameters"""
    with open_storage("storage/snake_oil/ensemble", "w") as storage:
        ensemble = storage.get_experiment_by_name(
            "ensemble-experiment"
        ).get_ensemble_by_name("default")

        ahmanalysis.AhmAnalysisJob().initializeAndRun(
            argument_types=[],
            argument_values=[None, None, "obs_key"],
            fixtures={
                "storage": storage,
                "ensemble": ensemble,
                "reports_dir": (
                    snake_oil_facade.config.analysis_config.log_path
                    / ensemble.experiment.name
                ),
                "observation_settings": snake_oil_facade.config.analysis_config.observation_settings,
                "es_settings": snake_oil_facade.config.analysis_config.es_settings,
                "random_seed": snake_oil_facade.config.random_seed,
            },
        )

    # assert that this returns/generates a KS csv file
    output_dir = Path("log/update/ensemble-experiment/AhmAnalysisJob")
    group_obs = [
        "All_obs",
        "All_obs-FOPR",
        "All_obs-WOPR_OP1_108",
        "All_obs-WOPR_OP1_144",
        "All_obs-WOPR_OP1_190",
        "All_obs-WOPR_OP1_36",
        "All_obs-WOPR_OP1_72",
        "All_obs-WOPR_OP1_9",
        "All_obs-WPR_DIFF_1",
        "FOPR",
        "WOPR_OP1_108",
        "WOPR_OP1_144",
        "WOPR_OP1_190",
        "WOPR_OP1_36",
        "WOPR_OP1_72",
        "WOPR_OP1_9",
        "WPR_DIFF_1",
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
    assert sorted(ks_df.columns[1:].tolist()) == sorted(group_obs)
    wopr_op1_df = ks_df[[c for c in ks_df.columns if c.startswith("WOPR_OP1")]]
    assert wopr_op1_df.max().max() <= 1
    assert wopr_op1_df.min().min() >= 0
    assert (output_dir / "active_obs_info.csv").is_file()
    assert (output_dir / "misfit_obs_info.csv").is_file()
    assert (output_dir / "prior.csv").is_file()
    for group in group_obs:
        filename = group + ".csv"
        assert (output_dir / filename).is_file()


@pytest.mark.integration_test
def test_ahmanalysis_run_deactivated_obs(copy_snake_oil_case_storage, snapshot, caplog):
    """
    We simulate a case where some of the observation groups are completely
    disabled by outlier detection
    """

    with Path("snake_oil.ert").open(mode="a", encoding="utf-8") as f:
        f.write("ENKF_ALPHA 0.1")

    snake_oil_facade = LibresFacade.from_config_file("snake_oil.ert")

    with (
        open_storage("storage/snake_oil/ensemble", "w") as storage,
        caplog.at_level(logging.WARNING),
    ):
        ensemble = storage.get_experiment_by_name(
            "ensemble-experiment"
        ).get_ensemble_by_name("default")

        ahmanalysis.AhmAnalysisJob().initializeAndRun(
            argument_types=[],
            argument_values=[None, None, "data_key"],
            fixtures={
                "storage": storage,
                "ensemble": ensemble,
                "reports_dir": (
                    snake_oil_facade.config.analysis_config.log_path
                    / ensemble.experiment.name
                ),
                "observation_settings": snake_oil_facade.config.analysis_config.observation_settings,
                "es_settings": snake_oil_facade.config.analysis_config.es_settings,
                "random_seed": snake_oil_facade.config.random_seed,
            },
        )

    assert "Analysis failed for" in caplog.text

    # assert that this returns/generates a KS csv file
    output_dir = Path("log/update/ensemble-experiment/AhmAnalysisJob")
    ks_df = pd.read_csv(output_dir / "ks.csv")
    snapshot.assert_match(ks_df.iloc[:10].to_csv(), "ks_df")


@pytest.mark.usefixtures("setup_tmpdir")
def test_that_dataset_with_no_prior_will_fail(test_data_root, capsys):
    test_data_dir = os.path.join(test_data_root, "snake_oil")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))
    snake_oil_facade = LibresFacade.from_config_file("snake_oil.ert")

    with open_storage("storage/snake_oil/ensemble", "w") as storage:
        ensemble = storage.create_experiment().create_ensemble(
            name="default", ensemble_size=10
        )

        ahmanalysis.AhmAnalysisJob().initializeAndRun(
            argument_types=[],
            argument_values=[],
            fixtures={
                "storage": storage,
                "ensemble": ensemble,
                "reports_dir": (
                    snake_oil_facade.config.analysis_config.log_path
                    / ensemble.experiment.name
                ),
                "observation_settings": snake_oil_facade.config.analysis_config.observation_settings,
                "es_settings": snake_oil_facade.config.analysis_config.es_settings,
                "random_seed": snake_oil_facade.config.random_seed,
            },
        )

        assert "Empty prior ensemble" in capsys.readouterr().err


def test_ahmanalysis_run_cli(snake_oil_facade):
    # Make workflow invoking ahmanalysis workflow job with args
    # add it to ert config
    # then run it

    with open("ahmanalysis_wf", "w", encoding="utf-8") as wf_file:
        wf_file.write("AHM_ANALYSIS analysis_case default")

    with open("snake_oil.ert", mode="a", encoding="utf-8") as f:
        f.write("LOAD_WORKFLOW ahmanalysis_wf")

    subprocess.run(
        [
            "ert",
            "workflow",
            "ahmanalysis_wf",
            "snake_oil.ert",
            "--ensemble",
            "default",
        ],
        check=False,
    )

    # assert that this returns/generates a KS csv file
    output_dir = Path("log/update/AhmAnalysisJob").resolve()
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
    assert sorted(ks_df.columns[1:].tolist()) == sorted(group_obs)
    assert ks_df["WOPR_OP1"].max() <= 1
    assert ks_df["WOPR_OP1"].min() >= 0
    assert (output_dir / "active_obs_info.csv").is_file()
    assert (output_dir / "misfit_obs_info.csv").is_file()
    assert (output_dir / "prior.csv").is_file()
    for group in group_obs:
        filename = group + ".csv"
        assert (output_dir / filename).is_file()
