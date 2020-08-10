import json
import logging
import os
import pytest
import sys

from semeio.communication import SemeioScript, SEMEIOSCRIPT_LOG_FILE

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


def test_semeio_script_publish(tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    namespace = "arbitrary_data"
    data = [1, 2, 3, 4, "This is a very important string"]

    class MySuperScript(SemeioScript):
        def run(self, *args):
            self.reporter.publish(namespace, data)

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    expected_outputfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, namespace + ".json"
    )

    with open(expected_outputfile) as f:
        published_data = json.load(f)

    assert [data] == published_data


def test_semeio_script_logging(tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    msg = "My logging msg"

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error(msg)

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    expected_logfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, SEMEIOSCRIPT_LOG_FILE
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1
    assert msg in loaded_log[0]


def assert_log(messages, log_file):
    with open(log_file) as f:
        log_data = f.readlines()

    assert len(messages) == len(log_data)
    for msg, log_entry in zip(messages, log_data):
        assert msg in log_entry


@pytest.mark.parametrize(
    "messages",
    [
        ("msg",),
        ("This is a message!", "And this is another one"),
        tuple("message_{}".format(idx) for idx in range(10)),
    ],
)
def test_semeio_script_multiple_logging(messages, tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    class MySuperScript(SemeioScript):
        def run(self, *args):
            posted_messages = []
            for msg in messages:
                logging.error(msg)
                posted_messages.append(msg)

                assert_log(posted_messages, expected_logfile)

    expected_logfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, SEMEIOSCRIPT_LOG_FILE
    )

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    assert_log(messages, expected_logfile)


def test_semeio_script_post_logging(tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    logging.error("A second message - not from MySuperScript")

    expected_logfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, SEMEIOSCRIPT_LOG_FILE
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_pre_logging(tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")

    my_super_script = MySuperScript(ert)

    logging.error("A message - not from MySuperScript")

    my_super_script.run()

    expected_logfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, SEMEIOSCRIPT_LOG_FILE
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_post_logging_exception(tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")
            raise AssertionError("Bazinga")

    my_super_script = MySuperScript(ert)
    try:
        my_super_script.run()
    except AssertionError:
        pass

    logging.error("A second message - not from MySuperScript")

    expected_logfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, SEMEIOSCRIPT_LOG_FILE
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_keyword_args(tmpdir):
    tmpdir.chdir()
    enspath = "enspath"
    ert = Mock(
        resConfig=Mock(
            return_value=Mock(model_config=Mock(getEnspath=Mock(return_value=enspath)))
        )
    )

    class MySuperScript(SemeioScript):
        def run(self, param_A, param_B):
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_A)
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_B)

    my_super_script = MySuperScript(ert)
    my_super_script.run(param_B="param_B", param_A="param_A")

    expected_outputfile = os.path.join(
        enspath, "reports", MySuperScript.__name__, SEMEIOSCRIPT_LOG_FILE
    )

    with open(expected_outputfile) as f:
        published_msgs = f.readlines()
    assert published_msgs[0] == "param_A\n"
    assert published_msgs[1] == "param_B\n"
