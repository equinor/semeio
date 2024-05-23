import subprocess

import ert.shared.hook_implementations
import pytest
from ert.shared.plugins.plugin_manager import ErtPluginContext

import semeio.hook_implementations.forward_models

DEFAULT_CONFIG = """
JOBNAME TEST

QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 1


NUM_REALIZATIONS 1
MIN_REALIZATIONS 1

FORWARD_MODEL {}({})
"""


@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.parametrize(
    "entry_point, options",
    [
        ("overburden_timeshift", "-c config.ots"),
        ("design2params", "not_int"),
        ("design_kw", "no_template"),
        ("gendata_rft", "--eclbase not_ecl"),
        ("fm_pyscal", "not_file"),
    ],
)
def test_console_scripts_exit_code(script_runner, entry_point, options):
    """Verify that console scripts return with non-zero exit codes for selected
    configurations.

    This nonzero returncode should in the subsequent test also force ERT to
    fail when the same script is called as a FORWARD_MODEL, without relying on
    ERTs TARGET_FILE mechanism for determining failure.
    """
    assert script_runner.run(entry_point, options).returncode != 0


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
    ],
)
@pytest.mark.usefixtures("setup_tmpdir")
def test_forward_model_error_propagation(forward_model, configuration, expected_error):
    """Assert that hard errors in forward models are also
    hard errors for ERT.

    An expected error message from the forward model is asserted
    captured in a specific stderr file.
    """
    config = DEFAULT_CONFIG.format(forward_model, configuration)
    with open("config.ert", "w", encoding="utf-8") as file:
        file.write(config)

    with pytest.raises(
        subprocess.CalledProcessError,
        match=r"Command.*ert.*returned non-zero exit status",
    ), ErtPluginContext(
        plugins=[
            semeio.hook_implementations.forward_models,
            ert.shared.hook_implementations,
        ]
    ):
        subprocess.run(["ert", "test_run", "config.ert", "--verbose"], check=True)
    with open(
        f"simulations/realization-0/iter-0/{forward_model}.stderr.0", encoding="utf-8"
    ) as fin:
        error = fin.read()
    assert expected_error in error
