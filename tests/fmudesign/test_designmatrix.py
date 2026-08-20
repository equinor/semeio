"""Tests for the DesignMatrix API and installed command."""

import shutil
import subprocess

import pandas as pd
import pytest

from semeio.fmudesign import DesignMatrix


def assert_valid_designmatrix(design_values):
    assert design_values.columns[:3].tolist() == ["REAL", "SENSNAME", "SENSCASE"]
    assert design_values["REAL"].tolist() == list(range(len(design_values)))
    assert not design_values.isna().any().any()


def test_designmatrix():
    design = DesignMatrix()
    design.generate(
        {
            "designtype": "onebyone",
            "seeds": "default",
            "repeats": 10,
            "distribution_seed": 42,
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
    )

    assert_valid_designmatrix(design.designvalues)
    assert len(design.designvalues) == 10
    assert isinstance(design.defaultvalues, dict)


@pytest.mark.integration_test
def test_endpoint_with_relative_input_and_custom_output_paths(
    tmp_path, monkeypatch, fmudesign_test_data
):
    source_design = fmudesign_test_data / "config/design_input_onebyone.xlsx"
    dependency = (
        pd.read_excel(source_design, header=None, engine="openpyxl")
        .set_index([0])[1]
        .to_dict()["background"]
    )

    case_dir = tmp_path / "path" / "going" / "down"
    case_dir.mkdir(parents=True)
    shutil.copy2(source_design, case_dir)
    shutil.copy2(source_design.parent / dependency, case_dir)
    monkeypatch.chdir(tmp_path)

    relative_design = case_dir.relative_to(tmp_path) / source_design.name
    output_path = tmp_path / "custom-design.xlsx"
    result = subprocess.run(
        ["fmudesign", str(relative_design), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reading file:" in result.stdout
    assert "Reading background values from:" in result.stdout
    assert "Adjusted to nearest consistent correlation matrix:" in result.stdout
    assert "Design matrix of shape (91, 22) written to:" in result.stdout
    assert "Thank you for using fmudesign" in result.stdout

    assert output_path.is_file()
    assert_valid_designmatrix(pd.read_excel(output_path, engine="openpyxl"))


@pytest.mark.integration_test
def test_endpoint_resolves_external_seeds_file_relative_to_input(
    tmp_path, monkeypatch, fmudesign_test_data
):
    """'rms_seeds' can also point to an external file. Like 'background' above,
    it must be resolved relative to the input file, not the CWD: the seeds file
    is only copied into the nested case_dir, so a CWD-relative fallback would
    fail to find it."""
    source_design = fmudesign_test_data / "config/design_input_background_extseeds.xlsx"

    case_dir = tmp_path / "path" / "going" / "down"
    case_dir.mkdir(parents=True)
    shutil.copy2(source_design, case_dir)
    shutil.copy2(source_design.parent / "seeds.xlsx", case_dir)
    shutil.copy2(source_design.parent / "doe1.xlsx", case_dir)
    monkeypatch.chdir(tmp_path)

    relative_design = case_dir.relative_to(tmp_path) / source_design.name
    output_path = tmp_path / "extseeds-design.xlsx"
    subprocess.run(
        ["fmudesign", str(relative_design), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    design_values = pd.read_excel(output_path, engine="openpyxl")
    assert_valid_designmatrix(design_values)
    # seeds.xlsx starts at 2000, unlike the 'default' 1000... sequence, so this
    # confirms the external file was actually read.
    assert design_values["RMS_SEED"].iloc[0] == 2000
