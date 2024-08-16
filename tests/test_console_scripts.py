import pytest


@pytest.mark.script_launch_mode("subprocess")
@pytest.mark.parametrize(
    "entry_point",
    [
        "csv_export2",
        "overburden_timeshift",
        "design2params",
        "gendata_rft",
        "design_kw",
        "fm_pyscal",
    ],
)
def test_console_scripts_help(script_runner, entry_point):
    """
    This test is a quick test that the scripts are installed correctly,
    by checking that we call call --help (which all these have). Semeio
    must be installed for them to pass.
    """
    ret = script_runner.run(entry_point, "--help")
    assert ret.success
