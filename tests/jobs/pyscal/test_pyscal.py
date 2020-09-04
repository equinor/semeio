"""Test module for ERT-Pyscal integration"""
import os
import sys
import random

import pytest

import pandas as pd

from semeio.jobs.scripts.fm_pyscal import run, main_entry_point


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
        (EXAMPLE_STATIC_DFRAME, ["__NONE__", "__NONE__", "__NONE__", "slgof", 2]),
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
    if random.randint(0, 1):
        genkw_prefix = "FOO:"
    else:
        genkw_prefix = ""

    if "CASE" in dframe:
        with open("parameters.txt", "w") as p_file:
            p_file.write("INTERPOLATE_WO 0.1\n" + genkw_prefix + "INTERPOLATE_GO 0.5")

    run(*(["relperm-input.csv", "relperm.inc"] + runargs))
    assert os.path.exists("relperm.inc")
    assert len(open("relperm.inc").readlines()) > 20


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
    assert "SGOF" in "".join(open("static.inc").readlines())
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
    assert "SGOF" not in "".join(open("wateroil.inc").readlines())


def test_fm_pyscal_argparse(tmpdir):
    """Test the command line wrapper"""
    tmpdir.chdir()
    EXAMPLE_STATIC_DFRAME.to_excel("relperm-input.xlsx")
    sys.argv = [
        "fm_pyscal",
        "relperm-input.xlsx",
        "relperm.inc",
        "",
        "__NONE__",
        "__NONE__",
        "slgof",
        "1",
    ]
    main_entry_point()
    assert os.path.exists("relperm.inc")


@pytest.mark.parametrize(
    "raises, dframe, runargs",
    [
        (
            # Writing to a file we don't have permission to write to
            (OSError, IOError),
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
        ),
        (
            # Reading a file that does not exist
            (ValueError, SystemExit),
            None,
            ["not-existing.csv", "relperm.inc", "", "__NONE__", "__NONE__", "sgof", 1],
        ),
        (
            # Specifying a sheet that does not exist
            (Exception, SystemExit),
            # Pyscal 0.4.0 raises an UnboundLocalError in this case, but with correct
            # error messages upfront. The exact exception might improve, so just assert
            # anything is thrown:
            EXAMPLE_STATIC_DFRAME,
            ["file.xlsx", "rel.inc", "foo_sheet", "__NONE__", "__NONE__", "sgof", 1],
        ),
        (
            # GasOil-interpolation without WaterOil, this is not supported:
            (ValueError, SystemExit),
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
        ),
        (
            # Non-existing parameter name
            (ValueError, SystemExit),
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
        ),
        (
            # Non-existing parameter name
            (ValueError, SystemExit),
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
        ),
        (
            # Wrong magic name for interpolation parameter
            (ValueError, SystemExit),
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
        ),
        (
            # Wrong sgof type
            (ValueError, SystemExit),
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
        ),
        (
            # Wrong family type
            (ValueError, SystemExit),
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
        ),
    ],
)
def test_fm_pyscal_errors(raises, dframe, runargs, tmpdir):
    """Test that fm_pyscal or pyscal gives correct errors to
    common misusages"""
    tmpdir.chdir()
    if dframe is not None:
        if runargs[0].endswith("csv"):
            dframe.to_csv(runargs[0])
        else:
            dframe.to_excel(runargs[0])
    if dframe is not None and "CASE" in dframe:
        with open("parameters.txt", "w") as p_file:
            p_file.write("INTERPOLATE_WO 0.1\nINTERPOLATE_GO 0.5")

    with pytest.raises(raises):
        run(*runargs)
