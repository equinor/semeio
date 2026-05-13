import pathlib
import subprocess
from pathlib import Path

import pytest

from .forward_models.pyscal.test_pyscal import EXAMPLE_STATIC_DFRAME

DEFAULT_CONFIG = """
JOBNAME TEST

QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 1


NUM_REALIZATIONS 1
MIN_REALIZATIONS 1

FORWARD_MODEL {}({})
"""


@pytest.mark.integration_test
@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.parametrize(
    ("entry_point", "options"),
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
    assert script_runner.run([entry_point, options]).returncode != 0


@pytest.mark.ert_integration
@pytest.mark.parametrize(
    ("forward_model", "configuration", "expected_error"),
    [
        ("OTS", "<CONFIG>=config.ots", "config.ots is not an existing file!"),
        ("DESIGN2PARAMS", "<IENS>=not_int", "invalid int value: 'not_int'"),
        (
            "DESIGN_KW",
            "<template_file>=no_template",
            " no_template is not an existing file!",
        ),
        ("GENDATA_RFT", "<ECLBASE>=not_ecl", "The path not_ecl.RFT does not exist"),
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
    pathlib.Path("config.ert").write_text(config, encoding="utf-8")

    with pytest.raises(
        subprocess.CalledProcessError,
        match=r"Command.*ert.*returned non-zero exit status",
    ):
        subprocess.run(
            ["ert", "test_run", "--disable-monitoring", "config.ert", "--verbose"],
            check=True,
        )
    error = pathlib.Path(
        f"simulations/realization-0/iter-0/{forward_model}.stderr.0"
    ).read_text(encoding="utf-8")
    assert expected_error in error


@pytest.mark.usefixtures("setup_tmpdir")
def test_pyscal_accepts_delta_s_argument():
    config = DEFAULT_CONFIG.format(
        "PYSCAL", "<PARAMETER_FILE>=<CONFIG_PATH>/relperm-input.csv, <DELTA_S>=0.1"
    )
    expected_line_count = 4 * (int(1 / 0.1) + 1)
    EXAMPLE_STATIC_DFRAME.to_csv("relperm-input.csv", index=False)
    pathlib.Path("config.ert").write_text(config, encoding="utf-8")
    subprocess.run(
        ["ert", "test_run", "--disable-monitoring", "config.ert", "--verbose"],
        check=True,
    )
    outputted_lines = (
        Path("simulations/realization-0/iter-0/relperm.inc")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert (
        len([line for line in outputted_lines if line and line[0] in {"0", "1"}])
        == expected_line_count
    )
