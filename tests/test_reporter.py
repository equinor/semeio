import pytest

from semeio.reporter import Reporter


@pytest.mark.parametrize(
    "test_input, expected", [("test_value", "test_value"), (32, "32")]
)
def test_reporter_write(setup_tmpdir, test_input, expected):
    reporter = Reporter("job")
    reporter.report(key="key", value=test_input, write=True)

    with open("job_key.txt") as fh:
        lines = fh.readlines()

    assert lines == [expected]


def test_reporter_print(capsys):
    reporter = Reporter("job")
    reporter.report(key="key", value="test")

    captured = capsys.readouterr()
    assert captured.out == "test\n"
