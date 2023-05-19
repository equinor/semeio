import subprocess
from pathlib import Path

import pytest

ERT_CONFIG = """
RUNPATH realization-%d/iter-%d
JOBNAME TEST
QUEUE_SYSTEM LOCAL
NUM_REALIZATIONS 1
FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/file.txt, <TO>=file.txt)
FORWARD_MODEL REPLACE_STRING(<FROM>={FROM}, <TO>="{TO}", <FILE>=file.txt)
"""


@pytest.mark.parametrize(
    "input_text, replace_from, replace_to, expected",
    [
        ("something", "something", "else", "else"),
        (
            "I got some/thing to tell you",
            "some/thing",
            "a story",
            "I got a story to tell you",
        ),
    ],
)
def test_replace_string(tmpdir, input_text, replace_from, replace_to, expected):
    with tmpdir.as_cwd():
        ert_config_fname = "test.ert"
        Path(ert_config_fname).write_text(
            ERT_CONFIG.format(FROM=replace_from, TO=replace_to), encoding="utf-8"
        )

        Path("file.txt").write_text(input_text, encoding="utf-8")

        subprocess.run(["ert", "test_run", ert_config_fname], check=True)

        output = Path("realization-0/iter-0/file.txt").read_text(encoding="utf-8")

        assert output == expected
