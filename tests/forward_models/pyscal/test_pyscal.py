"""Test module for ERT-Pyscal integration"""

import os
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

from semeio.forward_models.scripts.fm_pyscal import main_entry_point, run

EXAMPLE_STATIC_DFRAME = pd.DataFrame(
    columns=["SATNUM", "Nw", "Now", "Nog", "Ng"],
    data=[[1, 2, 2, 3, 3], [2, 4, 4, 1, 1]],
)
EXAMPLE_SCAL = pd.DataFrame(
    columns=["SATNUM", "CASE", "Nw", "Now", "Nog", "Ng"],
    data=[[1, "Low", 2, 2, 3, 3], [1, "Base", 4, 4, 1, 1], [1, "High", 5, 5, 2, 2]],
)
EXAMPLE_WATEROIL = pd.DataFrame(columns=["SATNUM", "Nw", "NOW"], data=[[1, 2, 2]])


@pytest.mark.parametrize(
    "dframe, runargs",
    [
        (EXAMPLE_STATIC_DFRAME, ["__NONE__", "__NONE__", "__NONE__", "sgof", 1]),
        (EXAMPLE_STATIC_DFRAME, ["__NONE__", "__NONE__", "__NONE__", "slgof", 1]),
        pytest.param(
            EXAMPLE_STATIC_DFRAME,
            ["__NONE__", "__NONE__", "__NONE__", "slgof", 2],
            marks=pytest.mark.xfail(raises=SystemExit),
            id="SLGOF_family_2_not_meaningful",
        ),
        (EXAMPLE_STATIC_DFRAME, ["__NONE__", "__NONE__", "__NONE__", "sgof", 2]),
        (EXAMPLE_WATEROIL, ["__NONE__", "__NONE__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "INTERPOLATE_WO", "INTERPOLATE_GO", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "INTERPOLATE_WO", "INTERPOLATE_GO", "slgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "INTERPOLATE_WO", "INTERPOLATE_GO", "sgof", 2]),
        (EXAMPLE_SCAL, ["__NONE__", "INTERPOLATE_WO", "INTERPOLATE_WO", "sgof", 2]),
        (EXAMPLE_SCAL, ["__NONE__", "INTERPOLATE_WO", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__BASE__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__LOW__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__PESS__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__PESSIMISTIC__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__OPT__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__OPTIMISTIC__", "__NONE__", "sgof", 1]),
        (EXAMPLE_SCAL, ["__NONE__", "__BASE__", "__OPTIMISTIC__", "sgof", 1]),
    ],
)
def test_fm_pyscal(dframe, runargs, tmpdir):
    """Parametrized test function for fm_pyscal"""
    tmpdir.chdir()
    dframe.to_csv("relperm-input.csv", index=False)

    # Insert a genkw-prefix in some parameters.txt files:
    genkw_prefix = "FOO:" if random.randint(0, 1) else ""

    if "CASE" in dframe:
        Path("parameters.txt").write_text(
            f"INTERPOLATE_WO 0.1\n{genkw_prefix}INTERPOLATE_GO 0.5", encoding="utf-8"
        )

    run(*(["relperm-input.csv", "relperm.inc"] + runargs))
    assert os.path.exists("relperm.inc")
    assert len(Path("relperm.inc").read_text(encoding="utf-8")) > 20


def test_fm_pysal_static_xlsx(tmpdir):
    """Test fm_pyscal on a static xlsx file"""
    tmpdir.chdir()
    EXAMPLE_STATIC_DFRAME.to_excel("relperm-input.xlsx")
    run("relperm-input.xlsx", "relperm.inc", "", "__NONE__", "__NONE__", "slgof", 1)

    # pylint: disable=abstract-class-instantiated
    with pd.ExcelWriter("relperm-sheets.xlsx") as writer:
        EXAMPLE_STATIC_DFRAME.to_excel(writer, sheet_name="static")
        EXAMPLE_WATEROIL.to_excel(writer, sheet_name="wateroil")
    run(
        "relperm-sheets.xlsx", "static.inc", "static", "__NONE__", "__NONE__", "sgof", 1
    )
    assert os.path.exists("static.inc")
    assert "SGOF" in "".join(Path("static.inc").read_text(encoding="utf-8"))
    run(
        "relperm-sheets.xlsx",
        "wateroil.inc",
        "wateroil",
        "__NONE__",
        "__NONE__",
        "sgof",
        1,
    )
    assert os.path.exists("wateroil.inc")
    assert "SGOF" not in "".join(Path("wateroil.inc").read_text(encoding="utf-8"))


def test_fm_pyscal_argparse(tmpdir, monkeypatch):
    """Test the command line wrapper"""
    tmpdir.chdir()
    EXAMPLE_STATIC_DFRAME.to_excel("relperm-input.xlsx")
    arguments = [
        "fm_pyscal",
        "relperm-input.xlsx",
        "relperm.inc",
        "",
        "__NONE__",
        "__NONE__",
        "slgof",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main_entry_point()
    assert os.path.exists("relperm.inc")


@pytest.mark.parametrize(
    "dframe, runargs, err_str",
    [
        (
            # Writing to a file we don't have permission to write to
            EXAMPLE_STATIC_DFRAME,
            [
                "relperm-input.csv",
                "/relperm.inc",
                "",
                "__NONE__",
                "__NONE__",
                "sgof",
                1,
            ],
            "/relperm.inc",
        ),
        (
            # Reading a file that does not exist
            None,
            ["not-existing.csv", "relperm.inc", "", "__NONE__", "__NONE__", "sgof", 1],
            "not-existing",
        ),
        (
            # Specifying a sheet that does not exist
            # Pyscal 0.4.0 raises an UnboundLocalError in this case, but with correct
            # error messages upfront. The exact exception might improve, so just assert
            # anything is thrown:
            EXAMPLE_STATIC_DFRAME,
            ["file.xlsx", "rel.inc", "foo_sheet", "__NONE__", "__NONE__", "sgof", 1],
            "foo_sheet",  # Don't test on the exact error message from xlrd/pandas
        ),
        (
            # GasOil-interpolation without WaterOil, this is not supported:
            EXAMPLE_SCAL,
            [
                "scal-input.csv",
                "relperm.inc",
                None,
                "__NONE__",
                "INTERPOLATE_GO",
                "sgof",
                1,
            ],
            "WaterOil interpolation parameter missing",
        ),
        (
            # Non-existing parameter name
            EXAMPLE_SCAL,
            [
                "scal-input.csv",
                "relperm.inc",
                None,
                "FOOOO",
                "INTERPOLATE_GO",
                "sgof",
                1,
            ],
            "FOOOO not found in parameters.txt",
        ),
        (
            # Non-existing parameter name
            EXAMPLE_SCAL,
            [
                "scal-input.csv",
                "relperm.inc",
                None,
                "INTERPOLATE_WO",
                "INTERPOLATE_GOOOOOO",
                "sgof",
                1,
            ],
            "INTERPOLATE_GOOOOOO not found in parameters.txt",
        ),
        (
            # Wrong magic name for interpolation parameter
            EXAMPLE_SCAL,
            [
                "scal-input.csv",
                "relperm.inc",
                None,
                "__OPTIM__",
                "__NONE__",
                "sgof",
                1,
            ],
            "__OPTIM__ not found in parameters.txt",
        ),
        (
            # Wrong sgof type
            EXAMPLE_SCAL,
            [
                "scal-input.csv",
                "relperm.inc",
                None,
                "INTERPOLATE_WO",
                "INTERPOLATE_GO",
                "sgofffff",
                1,
            ],
            "Only supports sgof or slgof",
        ),
        (
            # Wrong family type
            EXAMPLE_SCAL,
            [
                "scal-input.csv",
                "relperm.inc",
                None,
                "INTERPOLATE_WO",
                "INTERPOLATE_GO",
                "sgof",
                3,
            ],
            "Family must be either 1 or 2",
        ),
        (
            # Wrong usage of brackets in parameter names
            EXAMPLE_SCAL,
            [
                "scal_input.csv",
                "relperm.inc",
                "__NONE__",
                "<INTERPOLATE_WO>",
                "INTERPOLATE_GO",
                "sgof",
                1,
            ],
            "Do not include brackets",
        ),
        (
            # Wrong usage of brackets in parameter names
            EXAMPLE_SCAL,
            [
                "scal_input.csv",
                "relperm.inc",
                "__NONE__",
                "INTERPOLATE_WO",
                "<INTERPOLATE_GO>",
                "sgof",
                1,
            ],
            "Do not include brackets",
        ),
    ],
)
def test_fm_pyscal_errors(dframe, runargs, err_str, tmpdir, caplog):
    """Test that fm_pyscal sys.exits and gives a correct error message
    to common mistakes"""
    tmpdir.chdir()
    if dframe is not None:
        if runargs[0].endswith("csv"):
            dframe.to_csv(runargs[0])
        else:
            dframe.to_excel(runargs[0])
    if dframe is not None and "CASE" in dframe:
        Path("parameters.txt").write_text(
            "INTERPOLATE_WO 0.1\nINTERPOLATE_GO 0.5", encoding="utf-8"
        )

    with pytest.raises(SystemExit):
        run(*runargs)

    assert err_str in caplog.text
