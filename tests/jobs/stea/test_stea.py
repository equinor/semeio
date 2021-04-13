import json
import sys
from unittest import mock
import pytest
from semeio.jobs.scripts import fm_stea
import shutil
import os
from stea import SteaResult, SteaKeys
import stea


TEST_STEA_PATH, _ = os.path.split(os.path.abspath(__file__))


@pytest.fixture()
def setup_stea(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    shutil.copytree(TEST_STEA_PATH, "stea")
    os.chdir(os.path.join("stea"))
    yield
    os.chdir(cwd)


def calculate_patch(stea_input):
    return SteaResult(
        {
            SteaKeys.KEY_VALUES: [
                {SteaKeys.TAX_MODE: SteaKeys.CORPORATE, SteaKeys.VALUES: {"NPV": 30}}
            ]
        },
        stea_input,
    )


@pytest.fixture(autouse=True)
def mock_project(monkeypatch):
    project = {
        SteaKeys.PROJECT_ID: 1,
        SteaKeys.PROJECT_VERSION: 1,
        SteaKeys.PROFILES: [
            {
                SteaKeys.PROFILE_ID: "a_very_long_string",
            },
        ],
    }
    mocked_project = mock.MagicMock(return_value=stea.SteaProject(project))
    monkeypatch.setattr(stea.SteaClient, "get_project", mocked_project)
    yield mocked_project


@pytest.fixture(autouse=True)
def mock_calculate(monkeypatch):
    mock_stea = mock.MagicMock()
    mock_stea.side_effect = calculate_patch
    monkeypatch.setattr(stea, "calculate", mock_stea)
    yield mock_stea


@pytest.mark.usefixtures("setup_stea")
def test_stea(mock_calculate, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script_name", "-c", "stea_input.yml"])
    fm_stea.main_entry_point()
    mock_calculate.assert_called_once()
    files = os.listdir(os.getcwd())
    # the resulting file i.e. key is defined in the input config file: stea_input.yml
    assert "NPV_0" in files
    assert "stea_response.json" in files


@pytest.mark.usefixtures("setup_stea")
def test_stea_response(monkeypatch):
    expected_result = {
        "response": [{"TaxMode": "Corporate", "Values": {"NPV": 30}}],
        "profiles": {"a_very_long_string": {"Id": "a_very_long_string"}},
    }
    monkeypatch.setattr(sys, "argv", ["script_name", "-c", "stea_input.yml"])
    fm_stea.main_entry_point()
    with open("stea_response.json", "r") as fin:
        result = json.load(fin)
    assert result == expected_result


@pytest.mark.usefixtures("setup_stea")
def test_stea_ecl_case_overwrite(monkeypatch):
    """
    We want to verify that the ecl_case argument is properly forwarded to stea, but
    there is no ecl_case, so we just assert that we fail with our custom file missing
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["script_name", "-c", "stea_input.yml", "--ecl_case", "custom_ecl_case"],
    )
    with pytest.raises(
        OSError, match="Failed to create summary instance from argument:custom_ecl_case"
    ):
        fm_stea.main_entry_point()


@pytest.mark.usefixtures("setup_stea")
def test_stea_default_ert_args(monkeypatch):
    """
    We want to verify that the default ert args are correct, so basically that the
    script does not fail.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script_name",
            "-c",
            "stea_input.yml",
            "-e",
            "__NONE__",
            "-r",
            "stea_response.json",
        ],
    )
    fm_stea.main_entry_point()
