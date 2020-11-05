import sys
import os
import shutil
import subprocess
import pytest

from ert_shared.plugins.plugin_manager import ErtPluginContext
import semeio.hook_implementations.jobs
import ert_shared.hook_implementations


@pytest.mark.skipif(sys.platform == "darwin", reason="Skip test for Mac OS")
@pytest.mark.parametrize(
    "nosim_command,data_input,data_expected",
    [
        (
            "INSERT_NOSIM",
            "RUNSPEC\n",
            "RUNSPEC\nNOSIM\n",
        ),
        (
            "INSERT_NOSIM",
            "RUNSPEC -- some comment about RUNSPEC and whatnot\n-- and more RUNSPEC\n",
            "RUNSPEC\nNOSIM\n-- and more RUNSPEC\n",
        ),
        (
            "INSERT_NOSIM",
            "RUNSPEC some comment about RUNSPEC and whatnot\n-- and more RUNSPEC\n",
            "RUNSPEC\nNOSIM\n-- and more RUNSPEC\n",
        ),
        (
            "REMOVE_NOSIM",
            "RUNSPEC\nNOSIM\n",
            "RUNSPEC\n",
        ),
        (
            "REMOVE_NOSIM",
            "RUNSPEC\nNOSIM\n-- and more RUNSPEC\n",
            "RUNSPEC\n-- and more RUNSPEC\n",
        ),
        (
            "REMOVE_NOSIM",
            (
                "RUNSPEC for some reason valid -- some comment RUNSPEC\n"
                "-- more\n"
                "NOSIM   more valid comment because it is after 8 chars"
                " -- and now a valid comment\n"
                "-- and a comment about NOSIM as well, not to be touched\n"
            ),
            (
                "RUNSPEC for some reason valid -- some comment RUNSPEC\n"
                "-- more\n-- and a comment about NOSIM as well, not to be touched\n"
            ),
        ),
    ],
)
def test_nosim(setup_tmpdir, nosim_command, data_input, data_expected):
    shutil.copy(os.path.join(os.path.dirname(__file__), "data", "nosim.ert"), ".")

    with open("nosim.ert", "a") as fh:
        fh.writelines(["FORWARD_MODEL {}".format(nosim_command)])

    with open("TEST.DATA", "w") as fh:
        fh.write(data_input)

    with ErtPluginContext(
        plugins=[semeio.hook_implementations.jobs, ert_shared.hook_implementations]
    ):
        subprocess.check_call(
            ["ert", "test_run", "nosim.ert", "--verbose"],
        )
    with open("nosim/real_0/iter_0/TEST.DATA") as fh:
        assert fh.read() == data_expected
