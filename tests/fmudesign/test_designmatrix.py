"""Testing generating design matrices from dictionary input"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from semeio.fmudesign import DesignMatrix

TESTDATA = Path(__file__).parent / "data"


def valid_designmatrix(dframe):
    """Performs general checks on a design matrix, that should always be valid"""
    assert "REAL" in dframe

    # REAL always starts at 0 and is consecutive
    assert dframe["REAL"][0] == 0
    assert dframe["REAL"].diff().dropna().unique() == 1

    assert "SENSNAME" in dframe.columns
    assert "SENSCASE" in dframe.columns

    # There should be no empty cells in the dataframe:
    assert not dframe.isna().sum().sum()


def test_designmatrix():
    """Test the DesignMatrix class"""

    design = DesignMatrix()

    mock_dict = {
        "designtype": "onebyone",
        "seeds": "default",
        "repeats": 10,
        "defaultvalues": {},
        "sensitivities": {
            "rms_seed": {"seedname": "RMS_SEED", "senstype": "seed", "parameters": None}
        },
    }

    design.generate(mock_dict)
    valid_designmatrix(design.designvalues)
    assert len(design.designvalues) == 10
    assert isinstance(design.defaultvalues, dict)


def test_endpoint(tmpdir, monkeypatch):
    """Test the installed endpoint

    Will write generated design matrices to the pytest tmpdir directory,
    usually /tmp/pytest-of-<username>/
    """
    designfile = TESTDATA / "config/design_input_onebyone.xlsx"

    # The xlsx file contains a relative path, relative to the input design sheet:
    dependency = (
        pd.read_excel(designfile, header=None, engine="openpyxl")
        .set_index([0])[1]
        .to_dict()["background"]
    )

    tmpdir.chdir()
    monkeypatch.chdir(tmpdir)
    # Copy over input files:
    shutil.copy(str(designfile), ".")
    shutil.copy(Path(designfile).parent / dependency, ".")

    result = subprocess.run(
        ["fmudesign", str(designfile)], check=True, capture_output=True, text=True
    )

    expected_output = """Added sensitivity : seed
    Added sensitivity : faults
    Added sensitivity : velmodel
    Added sensitivity : contacts
    Added sensitivity : multz
    Added sensitivity : sens6

    Warning: Correlation matrix is not consistent
    Requirements:
    - Ones on the diagonal
    - Positive semi-definite matrix

    Input correlation matrix:
    [[1.00 0.90 0.00 0.00]
    [0.90 1.00 0.90 0.00]
    [0.00 0.90 1.00 0.00]
    [0.00 0.00 0.00 1.00]]

    Adjusted to nearest consistent correlation matrix:
    [[1.00 0.74 0.11 0.00]
    [0.74 1.00 0.74 0.00]
    [0.11 0.74 1.00 0.00]
    [0.00 0.00 0.00 1.00]]

    Added sensitivity : sens7
    Added sensitivity : sens8
    Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM13. Will be filled with default values.
    Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM14. Will be filled with default values.
    Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM15. Will be filled with default values.
    Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM16. Will be filled with default values.
    A total of 91 realizations were generated
    Designmatrix written to generateddesignmatrix.xlsx"""

    assert result.stdout.split() == expected_output.split()
    assert Path("generateddesignmatrix.xlsx").exists  # Default output file
    valid_designmatrix(pd.read_excel("generateddesignmatrix.xlsx", engine="openpyxl"))

    subprocess.run(["fmudesign", str(designfile), "anotheroutput.xlsx"], check=True)
    assert Path("anotheroutput.xlsx").exists
