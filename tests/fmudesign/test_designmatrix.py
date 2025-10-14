"""Testing generating design matrices from dictionary input"""

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from semeio.fmudesign import DesignMatrix

TESTDATA = Path(__file__).parent / "data"


def matches(pattern: str, text: str) -> bool:
    """Match text against a pattern where <ANY> acts as a wildcard.

    Examples
    --------
    >>> matches("my name is <ANY>!", "my name is John!")
    True
    >>> matches("my <ANY> is <ANY>!", "my name is John!")
    True
    matches("my <ANY> is <ANY>!", "my name are John!")
    False
    """
    regex_pattern = re.escape(pattern)
    regex_pattern = regex_pattern.replace("<ANY>", ".+?")
    regex_pattern = f"^{regex_pattern}$"
    return bool(re.match(regex_pattern, text))


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
            "rms_seed": {
                "seedname": "RMS_SEED",
                "senstype": "seed",
                "parameters": None,
                "dependencies": {},
            }
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

    # Use <ANY> in the string below to match anything in CLI output
    expected_output = """Reading background values from: <ANY>doe1.xlsx
Generating sensitivity : seed
Added sensitivity : seed
Generating sensitivity : faults
Added sensitivity : faults
Generating sensitivity : velmodel
Added sensitivity : velmodel
Generating sensitivity : contacts
Added sensitivity : contacts
Generating sensitivity : multz
Wrote 10 samples from 'MULTZ_ILE'
Added sensitivity : multz
Generating sensitivity : sens6
Wrote 10 samples from 'PARAM5'
Wrote 10 samples from 'PARAM6'
Wrote 10 samples from 'PARAM7'
Wrote 10 samples from 'FAULT_SEAL'
Added sensitivity : sens6
Generating sensitivity : sens7
Sampling parameters in 'corr1': ['PARAM9', 'PARAM10', 'PARAM11', 'PARAM12']

Warning: Correlation matrix 'corr1' is inconsistent
Requirements:
  - All diagonal elements must be 1
  - All elements must be between -1 and 1
  - The matrix must be positive semi-definite

Input correlation matrix:
|         |   PARAM9 | PARAM10   | PARAM11   | PARAM12   |
|:--------|---------:|:----------|:----------|:----------|
| PARAM9  |     1.00 |           |           |           |
| PARAM10 |     0.90 | 1.00      |           |           |
| PARAM11 |     0.00 | 0.90      | 1.00      |           |
| PARAM12 |     0.00 | 0.00      | 0.00      | 1.00      |

Adjusted to nearest consistent correlation matrix:
|         |   PARAM9 | PARAM10   | PARAM11   | PARAM12   |
|:--------|---------:|:----------|:----------|:----------|
| PARAM9  |     1.00 |           |           |           |
| PARAM10 |     0.74 | 1.00      |           |           |
| PARAM11 |     0.11 | 0.74      | 1.00      |           |
| PARAM12 |     0.00 | 0.00      | 0.00      | 1.00      |
Wrote 30 samples from 'PARAM9'
Wrote 30 samples from 'PARAM10'
Wrote 30 samples from 'PARAM11'
Wrote 30 samples from 'PARAM12'
Added sensitivity : sens7
Generating sensitivity : sens8
Added sensitivity : sens8
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM13. Will be filled with default values.
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM14. Will be filled with default values.
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM15. Will be filled with default values.
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM16. Will be filled with default values.
A total of 91 realizations were generated
Designmatrix written to generateddesignmatrix.xlsx

 Thank you for using fmudesign <ANY>
 Documentation: https://equinor.github.io/fmu-tools/fmudesign.html
 Issues/bugs/feature requests: https://github.com/equinor/semeio/issues"""

    for stdout_line, expected_line in zip(
        result.stdout.split(), expected_output.split(), strict=False
    ):
        assert matches(expected_line, stdout_line)

    assert Path("generateddesignmatrix.xlsx").exists  # Default output file
    valid_designmatrix(pd.read_excel("generateddesignmatrix.xlsx", engine="openpyxl"))

    subprocess.run(["fmudesign", str(designfile), "anotheroutput.xlsx"], check=True)
    assert Path("anotheroutput.xlsx").exists
