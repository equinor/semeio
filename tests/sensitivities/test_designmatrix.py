"""Testing generating design matrices from dictionary input"""

import shutil
from pathlib import Path
import subprocess
from packaging import version

import pandas as pd

import pytest

from fmu.tools.sensitivities import DesignMatrix

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


@pytest.mark.skipif(
    version.parse(pd.__version__) < version.parse("0.25.0"),
    reason="Pandas 0.25.0 is required for fmudesign",
)
def test_designmatrix():
    """Test the DesignMatrix class"""

    design = DesignMatrix()

    mock_dict = dict(
        designtype="onebyone",
        seeds="default",
        repeats=10,
        defaultvalues=dict(),
        sensitivities=dict(
            rms_seed=dict(seedname="RMS_SEED", senstype="seed", parameters=None)
        ),
    )

    design.generate(mock_dict)
    valid_designmatrix(design.designvalues)
    assert len(design.designvalues) == 10
    assert isinstance(design.defaultvalues, dict)


@pytest.mark.skipif(
    version.parse(pd.__version__) < version.parse("0.25.0"),
    reason="Pandas 0.25.0 is required for fmudesign",
)
def test_endpoint(tmpdir):
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
    # Copy over input files:
    shutil.copy(str(designfile), ".")
    shutil.copy(Path(designfile).parent / dependency, ".")

    subprocess.run(["fmudesign", str(designfile)], check=True)

    assert Path("generateddesignmatrix.xlsx").exists  # Default output file
    valid_designmatrix(pd.read_excel("generateddesignmatrix.xlsx", engine="openpyxl"))

    subprocess.run(["fmudesign", str(designfile), "anotheroutput.xlsx"], check=True)
    assert Path("anotheroutput.xlsx").exists
