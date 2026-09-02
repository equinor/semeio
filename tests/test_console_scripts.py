from importlib.metadata import entry_points

import pytest

EXPECTED_ENTRYPOINTS = {
    "csv_export2",
    "overburden_timeshift",
    "design2params",
    "gendata_rft",
    "design_kw",
    "fm_pyscal",
    "replace_string",
}


@pytest.mark.integration_test
def test_that_console_scripts_are_installed_and_their_help_command_returns_exit_code_0(
    monkeypatch,
):
    semeio_entry_points = [
        e
        for e in iter(entry_points(group="console_scripts"))
        if e.value.startswith("semeio.")
    ]

    semeio_entry_point_names = {e.name for e in semeio_entry_points}
    assert semeio_entry_point_names == EXPECTED_ENTRYPOINTS

    for entrypoint in semeio_entry_points:
        func = entrypoint.load()
        monkeypatch.setattr("sys.argv", [entrypoint.name, "--help"])
        with pytest.raises(SystemExit) as system_exit:
            func()
        assert system_exit.value.code == 0
