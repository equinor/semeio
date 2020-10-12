import os
import shutil
import cwrap
import pytest
import pandas as pd
import semeio.workflows.ahm_analysis.ahmanalysis as ahmanalysis


from res.enkf import EnKFMain, ResConfig
from res.enkf.export import (
    SummaryObservationCollector,
    GenDataObservationCollector,
)
from ecl import EclDataType
from ecl.eclfile import EclKW
from ecl.grid import EclGridGenerator
from ecl.util.util import RandomNumberGenerator


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
    output_path = "share/output_analysis/"
    output_ks = (
        "storage/snake_oil/runpath/" + output_path + "scalar_analysis_case/ks.csv"
    )
    assert os.path.isfile(output_ks)
    ks_df = pd.read_csv(output_ks)
    assert not ks_df.empty
    output_activeobs = output_ks.replace("ks.csv", "active_obs_info.csv")
    assert os.path.isfile(output_activeobs)
    output_misfitobs = output_ks.replace("ks.csv", "misfit_obs_info.csv")
    assert os.path.isfile(output_misfitobs)
    output_prior = output_ks.replace("ks.csv", "prior.csv")
    assert os.path.isfile(output_prior)


@pytest.mark.usefixtures("setup_tmpdir")
def test_ahmanalysis_run_field(test_data_root):
    """test data_set with scalar and Field parameters"""
    test_data_dir = os.path.join(test_data_root, "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))
    grid = EclGridGenerator.createRectangular((10, 12, 5), (1, 1, 1))
    rng = RandomNumberGenerator()
    rng.setState("ABCD6375ejascEFGHIJ")
    for iens in range(10):
        permx = EclKW("PERMX", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        permx.assign(rng.getDouble())

        poro = EclKW("PORO", grid.getGlobalSize(), EclDataType.ECL_FLOAT)
        poro.assign(rng.getDouble())

        if not os.path.isdir("fields"):
            os.makedirs("fields")

        with cwrap.open("fields/permx%d.grdecl" % iens, "w") as f:
            permx.write_grdecl(f)

        with cwrap.open("fields/poro%d.grdecl" % iens, "w") as f:
            poro.write_grdecl(f)

    res_config = ResConfig("snake_oil_field.ert")
    res_config.convertToCReference(None)
    ert = EnKFMain(res_config)

    ahmanalysis.AhmAnalysisJob(ert).run(prior_name="default")

    # assert that this returns/generates the delta field parameter
    gen_obs_list = GenDataObservationCollector.getAllObservationKeys(ert)
    summary_obs_list = SummaryObservationCollector.getAllObservationKeys(ert)
    obs_keys = gen_obs_list + summary_obs_list
    output_path = "share/output_analysis/"
    output_deltafield = (
        "storage/snake_oil_field/runpath/"
        + output_path
        + "field_analysis_case/delta_fieldPERMX.csv"
    )
    assert os.path.isfile(output_deltafield)
    delta_df = pd.read_csv(output_deltafield, index_col=0)
    assert len(delta_df.columns) == 8 + (len(obs_keys) * 2) + 1
    # check field parameter is present and not empty in the final KS matrix
    output_ks = (
        "storage/snake_oil_field/runpath/" + output_path + "scalar_analysis_case/ks.csv"
    )
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

    # check that it fails
    with pytest.raises(Exception):
        ahmanalysis.AhmAnalysisJob(ert).run(prior_name="default")
