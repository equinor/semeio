"""Example use cases for semeio.fmudesign."""

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from semeio.fmudesign import DesignMatrix, excel_to_dict
from semeio.fmudesign.fmudesignrunner import EXAMPLES

EXAMPLE_FILES = [example.filename for example in EXAMPLES]
TESTDATA = Path(__file__).parent / "data"
TEST_FILES = sorted((TESTDATA / "config").glob("design_input*.xlsx"))


def _run_cli(*args):
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        ["fmudesign", *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_prediction_rejection_sampled_ensemble(tmp_path):
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 3],
            ["rms_seeds", "default"],
            ["background", "hmrealizations.xlsx"],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"],
        data=[
            ["HMREAL", "-1"],
            ["ORAT", 6000],
            ["RESTARTPATH", "FOO"],
            ["HMITER", "-1"],
        ],
    )
    pd.DataFrame(
        columns=["RESTARTPATH", "HMREAL", "HMITER"],
        data=[
            ["/scratch/foo/2020a_hm3/", 31, 3],
            ["/scratch/foo/2020a_hm3/", 38, 3],
            ["/scratch/foo/2020a_hm3/", 54, 3],
        ],
    ).to_excel(tmp_path / "hmrealizations.xlsx")

    input_path = tmp_path / "designinput.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name="general_input", index=False, header=None
        )
        pd.DataFrame(
            columns=[
                "sensname",
                "numreal",
                "type",
                "param_name",
                "dist_name",
                "dist_param1",
                "dist_param2",
            ],
            data=[
                ["ref", None, "background", None],
                ["oil_rate", None, "dist", "ORAT", "uniform", 5000, 9000],
            ],
        ).to_excel(writer, sheet_name="design_input", index=False)
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)

    design = DesignMatrix()
    design.generate(excel_to_dict(input_path))

    assert set(design.designvalues["RESTARTPATH"]) == {"/scratch/foo/2020a_hm3/"}
    assert set(design.designvalues["HMITER"]) == {3}
    assert design.designvalues["REAL"].tolist() == list(range(6))
    assert design.designvalues["SENSNAME"].tolist() == ["ref"] * 3 + ["oil_rate"] * 3
    assert design.designvalues["HMREAL"].tolist() == [31, 38, 54] * 2


@pytest.mark.parametrize(
    "gen_input_sheet", ["general_input", "General_Input", "GENERALINPUT"]
)
def test_constant_distribution(tmp_path, gen_input_sheet):
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 1],
            ["rms_seeds", "default"],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"], data=[["a", 1.0]]
    )
    design_input = pd.DataFrame(
        columns=[
            "sensname",
            "numreal",
            "type",
            "param_name",
            "dist_name",
            "dist_param1",
        ],
        data=[["montecarlo", 100, "dist", "a", "const", 1.0]],
    )

    input_path = tmp_path / "designinput.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name=gen_input_sheet, index=False, header=None
        )
        design_input.to_excel(writer, sheet_name="designinput", index=False)
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)

    design = DesignMatrix()
    design.generate(excel_to_dict(input_path, gen_input_sheet="generalinput"))

    assert len(design.designvalues) == 100
    assert set(design.designvalues["a"]) == {1.0}
    assert set(design.designvalues["SENSNAME"]) == {"montecarlo"}


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "designfile", TEST_FILES, ids=[path.stem for path in TEST_FILES]
)
def test_all_input_files(tmp_path, monkeypatch, designfile):
    monkeypatch.chdir(tmp_path)
    for filename in designfile.parent.iterdir():
        if filename.is_file():
            shutil.copy2(filename, tmp_path)

    _run_cli(designfile.name)
    assert (tmp_path / "generateddesignmatrix.xlsx").is_file()


@pytest.mark.integration_test
@pytest.mark.parametrize("verbosity", [1, 2])
def test_cli_verbosity_levels(tmp_path, monkeypatch, verbosity):
    designfile = TESTDATA / "config/design_input_background.xlsx"
    monkeypatch.chdir(tmp_path)
    for filename in designfile.parent.iterdir():
        if filename.is_file():
            shutil.copy2(filename, tmp_path)

    _run_cli(designfile.name, *(["--verbose"] * verbosity))


@pytest.mark.integration_test
@pytest.mark.parametrize("designfile", EXAMPLE_FILES, ids=EXAMPLE_FILES)
def test_all_example_files_cmd_init(tmp_path, monkeypatch, designfile):
    monkeypatch.chdir(tmp_path)
    _run_cli("init", designfile)
    _run_cli("run", designfile)
