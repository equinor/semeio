import os

import pandas as pd
import pytest

from semeio.forward_models.rft.gendata_rft import _write_gen_data_files, _write_simdata


@pytest.mark.parametrize(
    "dataname, input_data, expected_result",
    [
        ("pressure", {"order": [1, 2], "pressure": [10.0, 20.0]}, ["10.0\n", "20.0\n"]),
        ("pressure", {"order": [1, 2]}, ["-1\n", "-1\n"]),
        ("sgas", {"order": [1, 2], "sgas": [0.1, 0.2]}, ["0.1\n", "0.2\n"]),
        ("sgas", {"order": [2, 1], "sgas": [0.2, 0.1]}, ["0.1\n", "0.2\n"]),
    ],
)
def test_write_simdata(tmpdir, dataname, input_data, expected_result):
    with tmpdir.as_cwd():
        dframe = pd.DataFrame(input_data)
        _write_simdata("some_file_name", dataname, dframe)

        with open("some_file_name", encoding="utf8") as fin:
            result = fin.readlines()
        assert result == expected_result


def test_write_gen_data_files_always_pressure(tmpdir):
    """Check that a file with pressure is always written, even it if is not
    in the dataframe.

    Saturation data is only outputted when present (might change in the future).
    """
    tmpdir.chdir()
    dframe = pd.DataFrame({"order": [1, 2], "is_active": [0, 0]})
    _write_gen_data_files(dframe, ".", "A-1", 0)
    pressure_file = "RFT_A-1_0"
    assert os.path.exists(pressure_file)
    with open(pressure_file, encoding="utf8") as f_handle:
        pressure_lines = f_handle.read().splitlines()
    assert pressure_lines == ["-1", "-1"]

    swat_file = "RFT_A-1_SWAT_0"
    assert not os.path.exists(swat_file)
