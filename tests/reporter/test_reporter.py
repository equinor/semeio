import pytest
import numpy as np
from semeio.reporter import report


def test_empty_key(tmpdir):
    with tmpdir.as_cwd(), pytest.raises(ValueError):
        report("", 0)


def test_key_is_path(tmpdir):
    with tmpdir.as_cwd(), pytest.raises(ValueError):
        report("/etc/motd", 0)


def test_key_invalid_type(tmpdir):
    with tmpdir.as_cwd(), pytest.raises(TypeError):
        report(b"hi", 0)


def test_key(tmpdir):
    with tmpdir.as_cwd():
        report("hi", 0)
        with open("hi.json") as f:
            assert f.read() == "0"


def test_json(tmpdir):
    data_dict = {"foo": "bar", "baz": [{"key": 1}, {"nøkkel": 2}]}
    data_str = '{"foo": "bar", "baz": [{"key": 1}, {"n\\u00f8kkel": 2}]}'

    with tmpdir.as_cwd():
        report("my_data", data_dict)
        with open("my_data.json") as f:
            assert f.read() == data_str


def test_numpy(tmpdir):
    mat = np.matrix([[1, 2, 3], [float("inf"), 3.14, -1]])
    str_ = "1.0,2.0,3.0\ninf,3.14,-1.0\n"

    with tmpdir.as_cwd():
        report("my_mat", mat)
        with open("my_mat.csv") as f:
            assert f.read() == str_


def test_invalid(tmpdir):
    with tmpdir.as_cwd(), pytest.raises(TypeError):
        report("fail", tmpdir)
