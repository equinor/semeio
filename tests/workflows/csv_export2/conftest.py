import os

import pytest

NORNE_DIR = os.path.join(os.path.dirname(__file__), "../../test_data/norne")


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
            runpath = os.path.join(f"realization-{real}", f"iter-{iteration}")

            os.makedirs(runpath, exist_ok=True)

            os.symlink(
                os.path.join(NORNE_DIR, "NORNE_ATW2013.UNSMRY"),
                os.path.join(runpath, f"NORNE_{real}.UNSMRY"),
            )
            os.symlink(
                os.path.join(NORNE_DIR, "NORNE_ATW2013.SMSPEC"),
                os.path.join(runpath, f"NORNE_{real}.SMSPEC"),
            )
            if parameters:
                with open(
                    os.path.join(runpath, "parameters.txt"), "w", encoding="utf-8"
                ) as p_fileh:
                    p_fileh.write(f"FOO 1{real}{iteration}")
            # Ensure fmu-ensemble does not complain on missing STATUS
            with open(os.path.join(runpath, "STATUS"), "w", encoding="utf-8") as file_h:
                file_h.write("a:b\na: 09:00:00 .... 09:00:01")

    with open("runpathfile", "w", encoding="utf-8") as file_h:
        for iteration in iters:
            for real in reals:
                runpath = os.path.join(f"realization-{real}", f"iter-{iteration}")
                file_h.write(f"{real:03d} {runpath} NORNE_{real} {iteration:03d}\n")


@pytest.fixture()
def norne_mocked_ensembleset(setup_tmpdir):
    # pylint: disable=unused-argument
    mock_norne_data(reals=[0, 1], iters=[0, 1], parameters=True)


@pytest.fixture()
def norne_mocked_ensembleset_noparams(setup_tmpdir):
    # pylint: disable=unused-argument
    mock_norne_data(reals=[0, 1], iters=[0, 1], parameters=False)


@pytest.fixture(name="setup_tmpdir")
def fixture_setup_tmpdir(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    yield
    os.chdir(cwd)
