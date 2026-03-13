import os
from pathlib import Path

import pytest

NORNE_DIR = Path(__file__).parent.parent.parent / "test_data" / "norne"


def mock_norne_data(reals, iters, parameters=True):
    # pylint: disable=consider-using-f-string
    """From a single UNSMRY file, produce arbitrary sized ensembles.

    Summary data will be equivalent over realizations, but the
    parameters.txt is made unique.

    Writes realization-*/iter-* file structure in cwd.

    Args:
        reals (list): integers with realization indices wanted
        iters (list): integers with iter indices wanted
        parameters (bool): Whether to write parameters.txt in each runpath
    """
    for real in reals:
        for iteration in iters:
            runpath = Path(f"realization-{real}") / f"iter-{iteration}"

            runpath.mkdir(exist_ok=True, parents=True)

            (runpath / f"NORNE_{real}.UNSMRY").symlink_to(
                NORNE_DIR / "NORNE_ATW2013.UNSMRY"
            )
            (runpath / f"NORNE_{real}.SMSPEC").symlink_to(
                NORNE_DIR / "NORNE_ATW2013.SMSPEC"
            )
            if parameters:
                (runpath / "parameters.txt").write_text(
                    f"FOO 1{real}{iteration}", encoding="utf-8"
                )
            # Ensure fmu-ensemble does not complain on missing STATUS
            (runpath / "STATUS").write_text(
                "a:b\na: 09:00:00 .... 09:00:01", encoding="utf-8"
            )

    with Path("runpathfile").open("w", encoding="utf-8") as file_h:
        for iteration in iters:
            for real in reals:
                runpath = Path(f"realization-{real}") / f"iter-{iteration}"
                file_h.write(f"{real:03d} {runpath} NORNE_{real} {iteration:03d}\n")


@pytest.fixture
def norne_mocked_ensembleset(setup_tmpdir):
    # pylint: disable=unused-argument
    mock_norne_data(reals=[0, 1], iters=[0, 1], parameters=True)


@pytest.fixture
def norne_mocked_ensembleset_noparams(setup_tmpdir):
    # pylint: disable=unused-argument
    mock_norne_data(reals=[0, 1], iters=[0, 1], parameters=False)


@pytest.fixture(name="setup_tmpdir")
def fixture_setup_tmpdir(tmpdir):
    cwd = Path.cwd()
    tmpdir.chdir()
    yield
    os.chdir(cwd)
