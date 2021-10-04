import json
import logging
import os
import pytest
from threading import Thread

from semeio.communication import SEMEIOSCRIPT_LOG_FILE, SemeioScript

from unittest.mock import Mock


def _ert_mock(ensemble_path="storage", user_case_name="case_name"):
    resconfig_mock = Mock()
    resconfig_mock.model_config.getEnspath.return_value = ensemble_path
    resconfig_mock.user_config_file = "config_name.ert"
    ert_mock = Mock()
    fs_mock = Mock()
    fs_mock.return_value.getCaseName.return_value = user_case_name
    ert_mock.getEnkfFsManager.return_value.getCurrentFileSystem = fs_mock
    ert_mock.resConfig.return_value = resconfig_mock
    return ert_mock


def test_semeio_script_publish(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "case_name"
    ert = _ert_mock(ensemble_path, user_case_name)

    namespace = "arbitrary_data"
    data = [1, 2, 3, 4, "This is a very important string"]

    class MySuperScript(SemeioScript):
        def run(self, *args):
            self.reporter.publish(namespace, data)

    my_super_script = MySuperScript(ert)
    my_super_script.run()  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        namespace + ".json",
    )

    with open(expected_outputfile, encoding="utf-8") as f:
        published_data = json.load(f)

    assert [data] == published_data


def test_semeio_script_logging(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    msg = "My logging msg"

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error(msg)

    my_super_script = MySuperScript(ert)
    my_super_script.run()  # pylint: disable=not-callable

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1
    assert msg in loaded_log[0]


def assert_log(messages, log_file):
    with open(log_file, encoding="utf-8") as f:
        log_data = f.readlines()

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
def test_semeio_script_multiple_logging(messages, tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
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

    my_super_script = MySuperScript(ert)
    my_super_script.run()  # pylint: disable=not-callable

    assert_log(messages, expected_logfile)


def test_semeio_script_post_logging(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")

    my_super_script = MySuperScript(ert)
    my_super_script.run()  # pylint: disable=not-callable

    logging.error("A second message - not from MySuperScript")

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_pre_logging(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")

    my_super_script = MySuperScript(ert)

    logging.error("A message - not from MySuperScript")

    my_super_script.run()  # pylint: disable=not-callable

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_concurrent_logging(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")
            thread = Thread(target=lambda: logging.error("External event."))
            thread.start()
            thread.join()

    MySuperScript(ert).run()  # pylint: disable=not-callable

    expected_logfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_logfile, encoding="utf-8") as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_post_logging_exception(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, *args):
            logging.error("A message from MySuperScript")
            raise AssertionError("Bazinga")

    my_super_script = MySuperScript(ert)
    try:
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

    with open(expected_logfile, encoding="utf-8") as f:
        loaded_log = f.readlines()

    assert len(loaded_log) == 1


def test_semeio_script_keyword_args(tmpdir):
    # pylint: disable=not-callable
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_A, param_B):
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_A)
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_B)

    my_super_script = MySuperScript(ert)
    my_super_script.run(param_B="param_B", param_A="param_A")

    expected_outputfile = os.path.join(
        "reports",
        "config_name",
        user_case_name,
        MySuperScript.__name__,
        SEMEIOSCRIPT_LOG_FILE,
    )

    with open(expected_outputfile, encoding="utf-8") as f:
        published_msgs = f.readlines()
    assert published_msgs[0] == "param_A\n"
    assert published_msgs[1] == "param_B\n"


def test_semeio_script_relative_report_dir(tmpdir):
    tmpdir.chdir()
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_A):
            self._reports_dir = "a_new_subdir"
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_A)

    my_super_script = MySuperScript(ert)
    my_super_script.run(param_A="param_A")  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        "a_new_subdir",
        SEMEIOSCRIPT_LOG_FILE,
    )
    with open(expected_outputfile, encoding="utf-8") as f:
        published_msgs = f.readlines()
    assert published_msgs[0] == "param_A\n"


def test_semeio_script_absolute_report_dir(tmpdir):
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_A):
            self._reports_dir = tmpdir
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_A)

    my_super_script = MySuperScript(ert)
    my_super_script.run(param_A="param_A")  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        tmpdir,
        SEMEIOSCRIPT_LOG_FILE,
    )
    with open(expected_outputfile, encoding="utf-8") as f:
        published_msgs = f.readlines()
    assert published_msgs[0] == "param_A\n"


def test_semeio_script_subdirs(tmpdir):
    ensemble_path, user_case_name = "storage", "config_file"
    ert = _ert_mock(ensemble_path, user_case_name)

    class MySuperScript(SemeioScript):
        def run(self, param_A):
            self._reports_dir = os.path.join(tmpdir, "sub_dir_1", "sub_dir_2")
            self.reporter.publish_msg(SEMEIOSCRIPT_LOG_FILE, param_A)

    my_super_script = MySuperScript(ert)
    my_super_script.run(param_A="param_A")  # pylint: disable=not-callable

    expected_outputfile = os.path.join(
        tmpdir,
        "sub_dir_1",
        "sub_dir_2",
        SEMEIOSCRIPT_LOG_FILE,
    )
    with open(expected_outputfile, encoding="utf-8") as f:
        published_msgs = f.readlines()
    assert published_msgs[0] == "param_A\n"
