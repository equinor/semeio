import subprocess
import pytest

ert_config = """
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
        with open(ert_config_fname, "w", encoding="utf-8") as file_h:
            file_h.write(ert_config.format(FROM=replace_from, TO=replace_to))

        with open("file.txt", "w", encoding="utf-8") as file_h:
            file_h.write(input_text)

        subprocess.run(["ert", "test_run", ert_config_fname], check=True)

        with open("realization-0/iter-0/file.txt") as output_file:
            output = output_file.read()

        assert output == expected
