import json
import logging
import os
from threading import Thread
from unittest.mock import Mock

import pytest

from semeio.communication import SEMEIOSCRIPT_LOG_FILE, SemeioScript, semeio_script

# pylint: disable=method-hidden,no-self-use, arguments-differ


def _ert_mock(monkeypatch, ensemble_path="storage", user_case_name="case_name"):
    facade = Mock()
    facade.enspath = ensemble_path
    facade.user_config_file = "config_name.ert"
    facade.get_current_case_name.return_value = user_case_name
    facade_mock = Mock()
    facade_mock.return_value = facade
    monkeypatch.setattr(semeio_script, "LibresFacade", facade_mock)


def test_semeio_script_publish(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    namespace = "arbitrary_data"
    data = [1, 2, 3, 4, "This is a very important string"]

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            self.reporter.publish(namespace, data)

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run()  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        namespace + ".json",
    )

    with open(expected_outputfile, encoding="utf-8") as file:
        published_data = json.load(file)

    assert [data] == published_data


def test_semeio_script_logging(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    msg = "My logging msg"

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            logging.error(msg)

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run()  # pylint: disable=not-callable

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as file:
        loaded_log = file.readlines()

    assert len(loaded_log) == 1
    assert msg in loaded_log[0]


def assert_log(messages, log_file):
    with open(log_file, encoding="utf-8") as file:
        log_data = file.readlines()

    assert len(messages) == len(log_data)
    for msg, log_entry in zip(messages, log_data):
        assert msg in log_entry


@pytest.mark.parametrize(
    "messages",
    [
        ("msg",),
        ("This is a message!", "And this is another one"),
        tuple(f"message_{idx}" for idx in range(10)),
    ],
)
def test_semeio_script_multiple_logging(monkeypatch, messages, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            posted_messages = []
            for msg in messages:
                logging.error(msg)
                posted_messages.append(msg)

                assert_log(posted_messages, expected_logfile)

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run()  # pylint: disable=not-callable

    assert_log(messages, expected_logfile)


def test_semeio_script_post_logging(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            logging.error("A message from MySuperScript")

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run()  # pylint: disable=not-callable

    logging.error("A second message - not from MySuperScript")

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as file:
        loaded_log = file.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_pre_logging(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            logging.error("A message from MySuperScript")

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()

    logging.error("A message - not from MySuperScript")

    my_super_script.run()  # pylint: disable=not-callable

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as file:
        loaded_log = file.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_concurrent_logging(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            logging.error("A message from MySuperScript")
            thread = Thread(target=lambda: logging.error("External event."))
            thread.start()
            thread.join()

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    MySuperScript().run()  # pylint: disable=not-callable

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as file:
        loaded_log = file.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_post_logging_exception(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            # pylint: disable=unused-argument
            logging.error("A message from MySuperScript")
            raise AssertionError("Bazinga")

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    try:  # noqa SIM105
        my_super_script.run()  # pylint: disable=not-callable
    except AssertionError:
        pass

    logging.error("A second message - not from MySuperScript")

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as file:
        loaded_log = file.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_keyword_args(monkeypatch, tmpdir):
    # pylint: disable=not-callable
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_a, param_b):
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_a)
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_b)

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run(param_b="param_b", param_a="param_a")

    expected_outputfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_outputfile, encoding="utf-8") as file:
        published_msgs = file.readlines()
    assert published_msgs[0] == "param_a\n"
    assert published_msgs[1] == "param_b\n"


def test_semeio_script_relative_report_dir(monkeypatch, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_a):
            self._reports_dir = "a_new_subdir"
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_a)

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run(param_a="param_a")  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        "a_new_subdir",
        SEMEIOSCRIPT_LOG_FILE,
    )
    with open(expected_outputfile, encoding="utf-8") as file:
        published_msgs = file.readlines()
    assert published_msgs[0] == "param_a\n"


def test_semeio_script_absolute_report_dir(monkeypatch, tmpdir):
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_a):
            self._reports_dir = tmpdir
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_a)

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run(param_a="param_a")  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        tmpdir,
        SEMEIOSCRIPT_LOG_FILE,
    )
    with open(expected_outputfile, encoding="utf-8") as file:
        published_msgs = file.readlines()
    assert published_msgs[0] == "param_a\n"


def test_semeio_script_subdirs(monkeypatch, tmpdir):
    ensemble_path, user_case_name = "storage", "default"
    _ert_mock(monkeypatch, ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_a):
            self._reports_dir = os.path.join(tmpdir, "sub_dir_1", "sub_dir_2")
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_a)

    ensemble_mock = Mock()
    ensemble_mock.name = user_case_name
    my_super_script = MySuperScript()
    my_super_script.run(param_a="param_a")  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        tmpdir,
        "sub_dir_1",
        "sub_dir_2",
        SEMEIOSCRIPT_LOG_FILE,
    )
    with open(expected_outputfile, encoding="utf-8") as file:
        published_msgs = file.readlines()
    assert published_msgs[0] == "param_a\n"
