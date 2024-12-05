"""Testing code for generation of design matrices"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import semeio
from semeio.fmudesign import DesignMatrix, excel2dict_design
from semeio.fmudesign.create_design import nearest_correlation_matrix

TESTDATA = Path(__file__).parent / "data"


def test_generate_onebyone(tmpdir):
    """Test generation of onebyone design"""

    inputfile = TESTDATA / "config/design_input_example1.xlsx"

    input_dict = excel2dict_design(inputfile)

    # Note that repeats are set to 10 in general_input sheet.
    # So, there are 10 rows for each senscase of type seed and scenario.
    # However, there are 20 rows for multz because numreal is set to 20 in designinput.
    rows_in_design_matrix = 80

    design = DesignMatrix()
    design.generate(input_dict)
    # Checking dimensions of design matrix
    assert design.designvalues.shape == (rows_in_design_matrix, 10)

    # Write to disk and check some validity
    tmpdir.chdir()
    design.to_xlsx("designmatrix.xlsx")
    assert Path("designmatrix.xlsx").exists
    diskdesign = pd.read_excel("designmatrix.xlsx", engine="openpyxl")

    assert (
        diskdesign.columns
        == [
            "REAL",
            "SENSNAME",
            "SENSCASE",
            "RMS_SEED",
            "FAULT_POSITION",
            "DC_MODEL",
            "OWC1",
            "OWC2",
            "OWC3",
            "MULTZ_ILE",
        ]
    ).all()
    assert (diskdesign["REAL"].values == np.arange(rows_in_design_matrix)).all()
    ensemble_size = 10
    sensname = (
        ["rms_seed"] * ensemble_size
        + ["faults"] * 2 * ensemble_size  # 2 senscases, east and west
        + ["velmodel"] * ensemble_size
        + ["contacts"] * 2 * ensemble_size  # 2 contacts, shallow and deep
        + ["multz"] * 20
    )
    assert (diskdesign["SENSNAME"] == sensname).all()
    # Sensitivites of type seed like rms_seed automatically get senscase p10_p90,
    # so that P10/P90 is calculated for the tornado plot.
    assert (
        diskdesign[diskdesign["SENSNAME"] == "rms_seed"]["SENSCASE"] == "p10_p90"
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "faults"]["SENSCASE"]
        == ["east"] * ensemble_size + ["west"] * ensemble_size
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "velmodel"]["SENSCASE"] == "alternative"
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "contacts"]["SENSCASE"]
        == ["shallow"] * ensemble_size + ["deep"] * ensemble_size
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "multz"]["SENSCASE"] == ["p10_p90"] * 20
    ).all()

    # When rms_seed is set to default it means that RMS_SEED numbers
    # 1000, 1001,... are used.
    # Note that for most senscases, RMS_SEED goes from 1000 to 1009,
    # but that it goes from 1000 to 1019 for multz because numreal
    # is set to 20 in the designinput sheet.
    assert (
        diskdesign["RMS_SEED"]
        == list(range(1000, 1000 + ensemble_size)) * 6 + list(range(1000, 1000 + 20))
    ).all()

    diskdefaults = pd.read_excel(
        "designmatrix.xlsx", sheet_name="DefaultValues", header=None, engine="openpyxl"
    )
    assert (diskdefaults.columns == [0, 1]).all()
    assert (
        diskdefaults.iloc[:, 0]
        == [
            "RMS_SEED",
            "FAULT_POSITION",
            "DC_MODEL",
            "OWC1",
            "OWC2",
            "OWC3",
            "MULTZ_ILE",
            "PARAM1",
            "PARAM2",
            "PARAM3",
            "PARAM4",
        ]
    ).all()

    diskdefaults = diskdefaults.set_index(0)

    # FAULT_POSITION has two senscases, east with value -1 and west with value 1,
    # so we expect ensemble_size number of rows with -1s and ensemble_size rows with 1s.
    # We expect the remaining rows to be set to the base value
    # set in the defaultvalues sheet.
    fault_position_base = diskdefaults.loc["FAULT_POSITION"].to_list()
    fault_position = (
        fault_position_base * ensemble_size
        + [-1] * ensemble_size
        + [1] * ensemble_size
        + fault_position_base * (rows_in_design_matrix - 3 * ensemble_size)
    )
    assert (diskdesign["FAULT_POSITION"] == fault_position).all()

    dc_model_base = diskdefaults.loc["DC_MODEL"].to_list()
    dc_model = (
        dc_model_base * 3 * ensemble_size
        + ["alternative"] * ensemble_size
        + dc_model_base * (rows_in_design_matrix - 4 * ensemble_size)
    )
    assert (diskdesign["DC_MODEL"] == dc_model).all()

    owc1_base = diskdefaults.loc["OWC1"].to_list()
    owc1 = (
        owc1_base * 4 * ensemble_size
        + [2600] * ensemble_size
        + [2700] * ensemble_size
        + owc1_base * (rows_in_design_matrix - 6 * ensemble_size)
    )
    assert (diskdesign["OWC1"] == owc1).all()

    owc2_base = diskdefaults.loc["OWC2"].to_list()
    owc2 = (
        owc2_base * 4 * ensemble_size
        + [2700] * ensemble_size
        + [2800] * ensemble_size
        + owc2_base * (rows_in_design_matrix - 6 * ensemble_size)
    )
    assert (diskdesign["OWC2"] == owc2).all()

    owc3_base = diskdefaults.loc["OWC3"].to_list()
    owc3 = (
        owc3_base * 4 * ensemble_size
        + [2800] * ensemble_size
        + [2900] * ensemble_size
        + owc3_base * (rows_in_design_matrix - 6 * ensemble_size)
    )
    assert (diskdesign["OWC3"] == owc3).all()

    # MULTZ_ILE contains random numbers so we won't test it here.

    diskmetadata = pd.read_excel(
        "designmatrix.xlsx", sheet_name="Metadata", engine="openpyxl"
    )

    assert (diskmetadata.columns == ["Description", "Value"]).all()
    assert diskmetadata["Description"].iloc[0] == "Created using semeio version:"
    assert diskmetadata["Value"].iloc[0] == semeio.__version__
    assert diskmetadata["Description"].iloc[1] == "Created on:"

    # For the timestamp, we can't check the exact value since it will differ
    # each time the test runs. Instead, we can verify it's a valid datetime string
    # by attempting to parse it
    timestamp = diskmetadata["Value"].iloc[1]
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        pytest.fail("Timestamp in Metadata sheet is not in expected format")


def test_generate_full_mc_snapshot(snapshot):
    """Test that full monte carlo design matrix generation remains consistent.

    This is a snapshot test that verifies the entire output of the design matrix
    generation process, including both the design values and default values.
    """
    # Setup
    inputfile = TESTDATA / "config/design_input_mc_with_correls.xlsx"
    input_dict = excel2dict_design(inputfile)
    design = DesignMatrix()

    # Generate the design matrix
    design.generate(input_dict)

    # Prepare data for snapshot comparison
    snapshot_dict = {
        "designvalues": design.designvalues.to_dict("records"),
        "defaultvalues": dict(design.defaultvalues),
    }

    # Serialize to string for snapshot comparison
    snapshot_str = json.dumps(
        snapshot_dict,
        indent=2,
        sort_keys=True,
    )

    # Verify against snapshot
    snapshot.assert_match(snapshot_str, "design_output_mc_with_correls.json")


def test_generate_full_mc(tmpdir):
    """Test generation of full monte carlo"""
    inputfile = TESTDATA / "config/design_input_mc_with_correls.xlsx"
    input_dict = excel2dict_design(inputfile)

    design = DesignMatrix()
    design.generate(input_dict)

    # Checking dimensions of design matrix
    assert design.designvalues.shape == (500, 16)

    # Write to disk and check some validity
    tmpdir.chdir()
    design.to_xlsx("designmatrix.xlsx")
    assert Path("designmatrix.xlsx").exists
    diskdesign = pd.read_excel(
        "designmatrix.xlsx", sheet_name="DesignSheet01", engine="openpyxl"
    )
    assert "REAL" in diskdesign
    assert "SENSNAME" in diskdesign
    assert "SENSCASE" in diskdesign
    assert not diskdesign.empty

    diskdefaults = pd.read_excel(
        "designmatrix.xlsx", sheet_name="DefaultValues", engine="openpyxl"
    )
    assert not diskdefaults.empty
    assert len(diskdefaults.columns) == 2

    # Make sure adding dependent discrete parameters works.
    disk_depends = pd.read_excel(inputfile, sheet_name="depend1", engine="openpyxl")
    df_merged = pd.merge(diskdesign, disk_depends, on="DATO", how="inner")
    assert (df_merged["DERIVED_PARAM1_x"] == df_merged["DERIVED_PARAM1_y"]).all()
    assert (df_merged["DERIVED_PARAM2_x"] == df_merged["DERIVED_PARAM2_y"]).all()

    # Check that variables are correlated using Pearson correlation
    # Using 95% confidence intervals for correlation coefficients.
    #
    # The confidence interval calculation assumes:
    #   - Large sample size (n > 30)
    #   - Bivariate normal distribution of variables
    #   - Linear relationship between variables
    # When these assumptions are violated (e.g. with skewed distributions),
    # the intervals become less reliable
    r_obj = stats.pearsonr(diskdesign["OWC1"], diskdesign["OWC2"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= 0.5 <= r_ci[1]

    r_obj = stats.pearsonr(diskdesign["OWC2"], diskdesign["OWC3"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= -0.7 <= r_ci[1]

    r_obj = stats.pearsonr(diskdesign["PARAM1"], diskdesign["PARAM2"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= 0 <= r_ci[1]

    # Using wide tolerance because the non-linear transformation between normal
    # and target distributions can alter correlation strength.
    assert np.isclose(
        stats.spearmanr(diskdesign["PARAM1"], diskdesign["PARAM3"])[0], 0.2, atol=0.1
    )

    # Check that we can add correlations to discrete variables
    assert np.isclose(
        stats.spearmanr(diskdesign["DATO"], diskdesign["NTG1"])[0], 0.8, atol=0.1
    )

    date_fractions = diskdesign["DATO"].value_counts(normalize=True)
    assert date_fractions.loc["2018-11-02"] == 0.3
    assert date_fractions.loc["2018-11-03"] == 0.4
    assert date_fractions.loc["2018-11-04"] == 0.3


def test_generate_background(tmpdir):
    inputfile = TESTDATA / "config/design_input_background.xlsx"
    input_dict = excel2dict_design(inputfile)
    source_file = TESTDATA / "config/doe1.xlsx"
    dest_file = tmpdir.join("doe1.xlsx")
    shutil.copy2(source_file, dest_file)

    with tmpdir.as_cwd():
        design = DesignMatrix()
        design.generate(input_dict)

        # Check that background parameters have same values in different sensitivies.
        background_params = ["PARAM17", "PARAM18", "PARAM19"]

        background_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "background", background_params
        ]
        velmodel_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "velmodel", background_params
        ]

        assert (background_vals.values == velmodel_vals.values).all()

        faults_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "faults", background_params
        ]
        contacts_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "contacts", background_params
        ]

        assert (faults_vals.values == contacts_vals.values).all()

        sens6 = design.designvalues[design.designvalues["SENSNAME"] == "sens6"]
        # PARAM5 ~ TruncatedNormal(3, 1, 1, 5)
        # PARAM6 ~ Uniform(0, 1)
        assert np.isclose(
            stats.spearmanr(sens6["PARAM5"], sens6["PARAM6"])[0],
            0.8,
            atol=0.1,
        )
        sens7 = design.designvalues[design.designvalues["SENSNAME"] == "sens7"]

        # PARAM9 and PARAM10 have a target correlation of 0.9 in the design config.
        # The input correlation matrix is not positive semi-definite and is
        # transformed to the closest positive semi-definite correlation matrix.
        # The new correlation coefficient is 0.8.
        # Using wide tolerance because the non-linear transformation between normal
        # and target distributions can alter correlation strength.
        assert np.isclose(
            stats.spearmanr(sens7["PARAM9"], sens7["PARAM10"])[0],
            0.8,
            atol=0.2,
        )

        assert np.isclose(
            stats.spearmanr(sens7["PARAM10"], sens7["PARAM11"])[0],
            0.8,
            atol=0.20,
        )


class TestNearestCorrelationMatrix:
    @pytest.mark.parametrize("variables", range(2, 100, 10))
    def test_nearest_correlation_matrix(self, variables):
        """Test that we can cholesky decompose the solution."""

        rng = np.random.default_rng(variables)

        # Create a correlation matrix
        observations = rng.normal(size=(variables * 2, variables))
        matrix = np.corrcoef(observations, rowvar=False)

        # Taking the cholesky decomposition should work just fine
        np.linalg.cholesky(matrix)

        # Mess it up
        matrix = matrix + rng.normal(size=matrix.shape, scale=0.1)
        matrix = matrix - np.identity(variables) * np.mean(np.diag(matrix))

        # Now the cholesky decomposition should fail
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(matrix)

        # Adjust the matrix to its nearest correlation matrix
        correlation_matrix = nearest_correlation_matrix(matrix)

        # Taking the cholesky decomposition should work now
        np.linalg.cholesky(correlation_matrix)

        # Diagonal entries should be 1.0 and the matrix should be symmetric
        assert np.allclose(np.diag(correlation_matrix), 1.0)
        assert np.allclose(correlation_matrix, correlation_matrix.T)

    def test_nearest_correlation_matrix_on_matlab_example(self):
        """These matrices are from the 'nearcorr' docs:
        https://www.mathworks.com/help/stats/nearcorr.html
        """
        # The matrix we want to adjust to become a correlation matrix
        A = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, -0.936],
                [0.0, 1.0, -0.55, -0.3645, -0.53],
                [0.0, -0.55, 1.0, -0.0351, 0.0875],
                [0.0, -0.3645, -0.0351, 1.0, 0.4557],
                [-0.936, -0.53, 0.0875, 0.4557, 1.0],
            ]
        )

        W = np.array(
            [
                [0.0, 1.0, 0.1, 0.15, 0.25],
                [1.0, 0.0, 0.05, 0.025, 0.15],
                [0.1, 0.05, 0.0, 0.25, 1.0],
                [0.15, 0.025, 0.25, 0.0, 0.25],
                [0.25, 0.15, 1.0, 0.25, 0.0],
            ]
        )

        matlab_Y = np.array(
            [
                [1.0, 0.0014, 0.0287, -0.0222, -0.8777],
                [0.0014, 1.0, -0.498, -0.7268, -0.4567],
                [0.0287, -0.498, 1.0, -0.0358, 0.0878],
                [-0.0222, -0.7268, -0.0358, 1.0, 0.4465],
                [-0.8777, -0.4567, 0.0878, 0.4465, 1.0],
            ]
        )

        # The smallest eigenvalue of A is -0.1244...
        # The smallest eigenvalue of Y is 1.088e-06
        Y = nearest_correlation_matrix(A, weights=W)

        # Matlab output has 4 digits, so atol is set to 1e-4 here
        assert np.allclose(Y, matlab_Y, atol=1e-4)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
