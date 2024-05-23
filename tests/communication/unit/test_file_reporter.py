import itertools
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from semeio.communication import FileReporter


@pytest.mark.parametrize(
    "msg",
    ["msg", "This is a message!", "And this is another one"],
)
def test_file_reporter_publish_msg_valid_msg(msg, tmpdir):
    tmpdir.chdir()
    reporter = FileReporter(os.getcwd())

    namespace = "my_msg.log"
    reporter.publish_msg(namespace, msg)

    with open(namespace, encoding="utf-8") as file:
        loaded_msg = file.readlines()

    assert (msg + "\n",) == tuple(loaded_msg)


@pytest.mark.parametrize(
    "messages",
    [
        ("msg",),
        ("msg", "This is a message!"),
        ("msg", "This is a message!", "And this is another one"),
    ],
)
def test_file_reporter_publish_msg_multiple_messages(messages, tmpdir):
    tmpdir.chdir()
    reporter = FileReporter(os.getcwd())

    namespace = "my_msg.log"

    for msg in messages:
        reporter.publish_msg(namespace, msg)

    with open(namespace, encoding="utf-8") as file:
        loaded_msg = file.readlines()

    assert tuple(msg + "\n" for msg in messages) == tuple(loaded_msg)


def test_file_reporter_publish_msg_multiple_namespaces(tmpdir):
    tmpdir.chdir()
    namespace1 = "namespace1"
    namespace2 = "namespace2"

    reporter = FileReporter(os.getcwd())

    msg1 = "This is a statement!"
    msg2 = "This is another a statement!"
    reporter.publish_msg(namespace1, msg1)
    reporter.publish_msg(namespace2, msg2)

    with open(namespace1, encoding="utf-8") as file:
        namespace1_msg = file.readlines()
    assert (msg1 + "\n",) == tuple(namespace1_msg)

    with open(namespace2, encoding="utf-8") as file:
        namespace2_msg = file.readlines()
    assert (msg2 + "\n",) == tuple(namespace2_msg)


@pytest.mark.parametrize(
    ("folder", "namespace"),
    list(
        itertools.product(
            ("", "monkey", os.path.join("a", "b")),
            ("namespace", "my_messages.log"),
        )
    ),
)
def test_file_reporter_publisg_msg_output_location(folder, namespace, tmpdir):
    tmpdir.chdir()
    reporter = FileReporter(os.path.join(os.getcwd(), folder))

    msg = "The most arbitrary message there is"
    reporter.publish_msg(namespace, msg)

    expected_output_file = os.path.join(os.getcwd(), folder, namespace)

    with open(expected_output_file, encoding="utf-8") as file:
        loaded_msg = file.readlines()

    assert (msg + "\n",) == tuple(loaded_msg)


@pytest.mark.parametrize(
    "namespace",
    (
        "fdsm/fdsfds",
        "/ds/",
        "/..",
    ),
)
def test_file_reporter_publish_msg_invalid_namespace(namespace, tmpdir):
    tmpdir.chdir()
    reporter = FileReporter(os.getcwd())

    with pytest.raises(ValueError) as err_info:
        reporter.publish_msg(namespace, "")

    assert "Namespace contains path separators" in str(err_info.value)


def test_file_reporter_publish_msg_invalid_output_dir(tmpdir):
    tmpdir.chdir()

    output_dir = os.path.join(os.getcwd(), "output")
    Path(output_dir).write_text("You shall not be a directory", encoding="utf-8")

    reporter = FileReporter(output_dir)
    with pytest.raises(ValueError) as err_info:
        reporter.publish_msg("namespace", "")

    assert "Expected output_dir to be a directory" in str(err_info.value)


@pytest.mark.parametrize(
    "data",
    (
        "",
        None,
        [1, 2, "cat"],
        {"a": [1.0, 2], "b": "something"},
    ),
)
def test_file_reporter_publish_valid_json(data, tmpdir):
    tmpdir.chdir()
    namespace = "data"

    reporter = FileReporter(os.getcwd())
    reporter.publish(namespace, data)

    loaded_data = json.loads(Path(namespace + ".json").read_text(encoding="utf-8"))

    assert loaded_data == [data]


def test_file_reporter_publish_multiple_json(tmpdir):
    tmpdir.chdir()
    namespace = "some_data"
    reporter = FileReporter(os.getcwd())

    data = [0, [0, 12], "Dear JSON"]
    for idx, data_elem in enumerate(data):
        reporter.publish(namespace, data_elem)

        loaded_data = json.loads(Path(namespace + ".json").read_text(encoding="utf-8"))

        assert loaded_data == data[: idx + 1]


def test_file_reporter_publish_multiple_namespaces(tmpdir):
    tmpdir.chdir()
    namespace1 = "namespace1"
    namespace2 = "namespace2"

    data1 = "This is message number 1"
    data2 = "This is yet another message"

    reporter = FileReporter(os.getcwd())

    reporter.publish(namespace1, data1)
    reporter.publish(namespace2, data2)

    loaded_data1 = json.loads(Path(namespace1 + ".json").read_text(encoding="utf-8"))
    assert loaded_data1 == [data1]

    loaded_data2 = json.loads(Path(namespace2 + ".json").read_text(encoding="utf-8"))
    assert loaded_data2 == [data2]


@pytest.mark.parametrize(
    ("folder", "namespace"),
    list(
        itertools.product(
            ("", "monkey", os.path.join("a", "b")),
            ("namespace", "my_messages.log"),
        )
    ),
)
def test_file_reporter_publish_output_location(folder, namespace, tmpdir):
    tmpdir.chdir()
    reporter = FileReporter(os.path.join(os.getcwd(), folder))

    data = [1, 2, 1, 2, "Testing the most arbitrary data there is"]
    reporter.publish(namespace, data)

    expected_output_file = os.path.join(os.getcwd(), folder, namespace) + ".json"

    loaded_data = json.loads(Path(expected_output_file).read_text(encoding="utf-8"))
    assert [data] == loaded_data


@pytest.mark.parametrize(
    "namespace",
    (
        "fdsm/fdsfds",
        "/ds/",
        "/..",
    ),
)
def test_file_reporter_publish_invalid_namespace(namespace, tmpdir):
    tmpdir.chdir()
    reporter = FileReporter(os.getcwd())

    with pytest.raises(ValueError) as err_info:
        reporter.publish(namespace, "")

    assert "Namespace contains path separators" in str(err_info.value)


def test_file_reporter_publish_invalid_output_dir(tmpdir):
    tmpdir.chdir()

    output_dir = os.path.join(os.getcwd(), "output")
    Path(output_dir).write_text("You shall not be a directory", encoding="utf-8")

    reporter = FileReporter(output_dir)
    with pytest.raises(ValueError) as err_info:
        reporter.publish("namespace", "")

    assert "Expected output_dir to be a directory" in str(err_info.value)


@pytest.mark.parametrize(
    "output_dir",
    (
        "fdsm/fdsfds",
        ".",
        "..",
        "reports",
    ),
)
def test_file_reporter_relative_output_dir(output_dir, tmpdir):
    tmpdir.chdir()
    with pytest.raises(ValueError) as err_info:
        FileReporter(output_dir)

    assert "Expected output_dir to be an absolute path" in str(err_info.value)


@pytest.mark.parametrize(
    "data",
    (pd.DataFrame([[1, 2, 3], [2, 3, 5]]),),
)
def test_file_reporter_publish_valid_csv(data, tmpdir):
    tmpdir.chdir()
    namespace = "data"

    reporter = FileReporter(os.getcwd())
    reporter.publish_csv(namespace, data)

    loaded_data = pd.read_csv(namespace + ".csv", index_col=0, header=0)

    assert (loaded_data.to_numpy() == data.to_numpy()).all()


def test_file_reporter_publish_invalid_csv(tmpdir):
    tmpdir.chdir()
    namespace = "data"
    data = "not a dataframe"

    reporter = FileReporter(os.getcwd())

    with pytest.raises(AttributeError):
        reporter.publish_csv(namespace, data)
