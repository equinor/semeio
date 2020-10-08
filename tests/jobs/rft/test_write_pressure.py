import pytest
import pandas as pd
from semeio.jobs.rft.gendata_rft import _write_pressure


@pytest.mark.parametrize(
    "input_data, expected_result",
    [
        ({"order": [1, 2], "pressure": [10.0, 20.0]}, ["10.0\n", "20.0\n"]),
        ({"order": [1, 2]}, ["-1\n", "-1\n"]),
    ],
)
def test_write_pressure(tmpdir, input_data, expected_result):
    df = pd.DataFrame(input_data)
    _write_pressure("some_file_name", df)

    with open("some_file_name", "r") as fin:
        result = fin.readlines()
    assert result == expected_result
