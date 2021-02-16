import json
import os
import pytest
import shutil
import subprocess
import sys
from semeio.communication import SEMEIOSCRIPT_LOG_FILE


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test-data")
ERT_INSTALLED = shutil.which("ert") is not None


@pytest.mark.skipif(not ERT_INSTALLED, reason="ERT is not installed")
def test_semeio_script_integration(tmpdir):
    tmpdir.chdir()

    shutil.copytree(TEST_DATA_DIR, "test_data")
    os.chdir(os.path.join("test_data"))

    ert_env = {
        env_var: os.environ[env_var]
        for env_var in ("PATH", "LD_LIBRARY_PATH")
        if env_var in os.environ
    }
    ert_env["PYTHONPATH"] = os.pathsep.join(map(os.path.realpath, sys.path))

    subprocess.check_call(
        ("ert", "workflow", "TEST_WORKFLOW", "config.ert"),
        env=ert_env,
    )

    # Assert that data was published correctly
    with open("reports/default/TestWorkflowJob/test_data.json") as f:
        reported_data = json.load(f)
    assert [list(range(10))] == reported_data

    # Assert that logs were forwarded correctly
    log_file = os.path.join("reports/default/TestWorkflowJob/", SEMEIOSCRIPT_LOG_FILE)
    with open(log_file) as f:
        log = f.readlines()
    assert len(log) == 1
    expected_log_msg = (
        "I finished without any problems - hence I'm not a failure after all!"
    )
    assert expected_log_msg in log[0]
