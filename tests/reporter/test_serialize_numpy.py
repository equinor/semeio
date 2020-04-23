from semeio.reporter.serialize_numpy import serialize_numpy
import numpy as np


def test_invalid(tmpdir):
    with tmpdir.as_cwd():
        assert not serialize_numpy("try", 0)


def test_array(tmpdir):
    arr = np.array([float("inf"), 3.14, -1])
    str_ = "inf\n3.14\n-1.0\n"

    with tmpdir.as_cwd():
        assert serialize_numpy("arr", arr)
        with open("arr.csv") as f:
            assert f.read() == str_


def test_matrix(tmpdir):
    mat = np.matrix([[1, 2, 3], [float("inf"), 3.14, -1]])
    str_ = "1.0,2.0,3.0\ninf,3.14,-1.0\n"

    with tmpdir.as_cwd():
        assert serialize_numpy("mat", mat)
        with open("mat.csv") as f:
            assert f.read() == str_
