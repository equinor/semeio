from __future__ import absolute_import
import pytest
from semeio.jobs.scripts.fm_stea import main_entry_point
import shutil
import os
from stea import SteaResult, SteaKeys, SteaInput

import sys

if sys.version_info >= (3, 3):
    from unittest.mock import patch
else:
    from mock import patch

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


@patch("stea.calculate")
@pytest.mark.usefixtures("setup_stea")
def test_stea(mock_stea):
    mock_stea.side_effect = calculate_patch
    main_entry_point(["-c", "stea_input.yml"])
    mock_stea.assert_called_once()
    files = os.listdir(os.getcwd())
    # the resulting file i.e. key is defined in the input config file: stea_input.yml
    assert "NPV_0" in files
