import sys
import subprocess
import pytest

from ert_shared.plugins.plugin_manager import ErtPluginContext
import semeio.hook_implementations.jobs
import ert_shared.hook_implementations


default_config = """
JOBNAME TEST

QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 1


NUM_REALIZATIONS 1
MIN_REALIZATIONS 1

FORWARD_MODEL {}({})
"""


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "Forward models have same name as executable and "
        "libres will find wrong exe on case insensitive system "
        "bug report here: https://github.com/equinor/libres/issues/1053"
    ),
)
@pytest.mark.parametrize(
    "forward_model, configuration, expected_error",
    [
        ("OTS", "<CONFIG>=config.ots", "config.ots is not an existing file!"),
        ("DESIGN2PARAMS", "<IENS>=not_int", "invalid int value: 'not_int'"),
        (
            "DESIGN_KW",
            "<template_file>=no_template",
            " no_template is not an existing file!",
        ),
        ("GENDATA_RFT", "<ECL_BASE>=not_ecl", "The path not_ecl.RFT does not exist"),
        ("PYSCAL", "<PARAMETER_FILE>=not_file", "not_file does not exist"),
        ("STEA", "<CONFIG>=not_a_file", "not_a_file is not an existing file!"),
    ],
)
def test_console_script_integration(
    setup_tmpdir, forward_model, configuration, expected_error
):
    config = default_config.format(forward_model, configuration)
    with open("config.ert", "w") as fh:
        fh.write(config)

    with ErtPluginContext(
        plugins=[semeio.hook_implementations.jobs, ert_shared.hook_implementations]
    ):
        subprocess.run(["ert", "test_run", "config.ert", "--verbose"], check=False)
    with open(f"simulations/realization0/{forward_model}.stderr.0") as fin:
        error = fin.read()
    assert expected_error in error
