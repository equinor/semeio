import os
import shutil
import subprocess
import sys

import ert.shared.hook_implementations
import pytest
from ert.shared.plugins.plugin_manager import ErtPluginContext

import semeio.hook_implementations.forward_models


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Skip test for Mac OS - invalid use of sed"
)
# The GNU extension for the sed command on osx behaves slightly different.
# The result when running the forward model is as follow:
# cat insert_nosim.stderr.1
# sed: 1: "TEST.DATA": invalid command code T
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
@pytest.mark.usefixtures("setup_tmpdir")
def test_nosim(nosim_command, data_input, data_expected):
    shutil.copy(os.path.join(os.path.dirname(__file__), "data", "nosim.ert"), ".")

    with open("nosim.ert", "a", encoding="utf-8") as file:
        file.writelines([f"FORWARD_MODEL {nosim_command}"])

    with open("TEST.DATA", "w", encoding="utf-8") as file:
        file.write(data_input)

    with ErtPluginContext(
        plugins=[
            semeio.hook_implementations.forward_models,
            ert.shared.hook_implementations,
        ]
    ):
        subprocess.check_call(
            ["ert", "test_run", "nosim.ert", "--verbose"],
        )
    with open("nosim/realization-0/iter-0/TEST.DATA", encoding="utf-8") as file:
        assert file.read() == data_expected
