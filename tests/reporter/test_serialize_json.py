from semeio.reporter.serialize_json import serialize_json


def test_simple(tmpdir):
    with tmpdir.as_cwd():
        assert serialize_json("test", "Test")
        with open("test.json") as f:
            assert f.read() == '"Test"'


def test_complex(tmpdir):
    data_dict = {"foo": "bar", "baz": [{"key": 1}, {"nøkkel": 2}]}
    data_str = '{"foo": "bar", "baz": [{"key": 1}, {"n\\u00f8kkel": 2}]}'

    with tmpdir.as_cwd():
        assert serialize_json("complex", data_dict)
        with open("complex.json") as f:
            assert f.read() == data_str


def test_simple_fail(tmpdir):
    # tmpdir object is not JSON-serialisable
    with tmpdir.as_cwd():
        assert not serialize_json("test", tmpdir)


def test_complex_fail(tmpdir):
    data_dict = {"foo": "bar", "baz": [{"key": serialize_json}, {"nøkkel": 2}]}
    with tmpdir.as_cwd():
        assert not serialize_json("test", data_dict)


def test_recursive_fail(tmpdir):
    data = {}
    data["data"] = data
    with tmpdir.as_cwd():
        assert not serialize_json("rec", data)
