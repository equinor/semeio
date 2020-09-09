import json
import logging
import os
import pytest
from threading import Thread

from semeio.communication import SEMEIOSCRIPT_LOG_FILE, SemeioScript

from unittest.mock import Mock


def _ert_mock(config_path="config_path", user_config_file="config_file"):
    resconfig_mock = Mock()
    resconfig_mock.config_path = config_path
    resconfig_mock.user_config_file = user_config_file

    return Mock(resConfig=Mock(return_value=resconfig_mock))


def test_semeio_script_publish(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    namespace = "arbitrary_data"
    data = [1, 2, 3, 4, "This is a very important string"]

    class MySuperScript(SemeioScript):
        def run(self, *args):
            self.reporter.publish(namespace, data)

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    expected_outputfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        namespace + ".json",
    )

    with open(expected_outputfile) as f:
        published_data = json.load(f)

    assert [data] == published_data


def test_semeio_script_logging(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    msg = "My logging msg"

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error(msg)

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    expected_logfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
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
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            posted_messages = []
            for msg in messages:
                logging.error(msg)
                posted_messages.append(msg)

                assert_log(posted_messages, expected_logfile)

    expected_logfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    assert_log(messages, expected_logfile)


def test_semeio_script_post_logging(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")

    my_super_script = MySuperScript(ert)
    my_super_script.run()

    logging.error("A second message - not from MySuperScript")

    expected_logfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_pre_logging(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")

    my_super_script = MySuperScript(ert)

    logging.error("A message - not from MySuperScript")

    my_super_script.run()

    expected_logfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_concurrent_logging(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")
            thread = Thread(target=lambda: logging.error("External event."))
            thread.start()
            thread.join()

    MySuperScript(ert).run()

    expected_logfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_post_logging_exception(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

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
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile) as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_keyword_args(tmpdir):
    tmpdir.chdir()
    config_path, user_config_file = "config_path", "config_file"
    ert = _ert_mock(config_path, user_config_file)

    class MySuperScript(SemeioScript):
        def run(self, param_A, param_B):
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_A)
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_B)

    my_super_script = MySuperScript(ert)
    my_super_script.run(param_B="param_B", param_A="param_A")

    expected_outputfile = os.path.join(
        config_path,
        "reports",
        user_config_file,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_outputfile) as f:
        published_msgs = f.readlines()
    assert published_msgs[0] == "param_A\n"
    assert published_msgs[1] == "param_B\n"
