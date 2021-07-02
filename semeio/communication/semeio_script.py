import datetime
import logging
import os
import threading
from pathlib import Path
from logging.handlers import BufferingHandler
from types import MethodType

from res.enkf import ErtScript

from semeio.communication.reporter import FileReporter

SEMEIOSCRIPT_LOG_FILE = "workflow-log.txt"


class _LogHandlerContext(object):
    def __init__(self, log, handler):
        self._log = log
        self._handler = handler

    def __enter__(self):
        self._log.addHandler(self._handler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.removeHandler(self._handler)


class _ReportHandler(BufferingHandler):
    def __init__(self, output_dir, thread_id):
        super().__init__(1)
        self._reporter = FileReporter(output_dir)
        self._namespace = SEMEIOSCRIPT_LOG_FILE
        self.addFilter(lambda rec: rec.thread == thread_id)

    def flush(self):
        for log_record in self.buffer:
            self._reporter.publish_msg(self._namespace, self._format_record(log_record))

        super().flush()

    def _format_record(self, log_record):
        log_fmt = "{log_level} [{log_time}]: {log_message}"
        return log_fmt.format(
            log_level=log_record.levelname,
            log_time=datetime.datetime.fromtimestamp(log_record.created),
            log_message=log_record.message,
        )


class SemeioScript(ErtScript):  # pylint: disable=too-few-public-methods
    """
    SemeioScript is a workflow utility extending the functionality of ErtScript.
    In particular it provides a `self.reporter` instance available for passing
    data to the common storage. In addition, while `self.run` is being executed
    it forwards log statements to the reporter as well.
    """

    def __init__(self, ert):
        super().__init__(ert)
        self._output_dir = self._get_output_dir()
        self._reporter = FileReporter(self._output_dir)
        self._wrap_run()

    def _wrap_run(self):
        # pylint: disable=access-member-before-definition
        self._real_run = self.run

        def run_with_handler(self, *args, **kwargs):
            log = logging.getLogger("")
            thread_id = threading.get_ident()
            report_handler = _ReportHandler(self._output_dir, thread_id)
            with _LogHandlerContext(log, report_handler):
                self._real_run(*args, **kwargs)

        self.run = MethodType(run_with_handler, self)

    def _get_output_dir(self):
        res_config = self.ert().resConfig()
        base_dir = Path(res_config.model_config.getEnspath()).parent.absolute()
        sub_dir = str(
            self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()
        )
        return os.path.join(
            base_dir,
            "reports",
            Path(res_config.user_config_file).stem,
            sub_dir,
            type(self).__name__,
        )

    @property
    def reporter(self):
        return self._reporter

    @property
    def _reports_dir(self):
        return self.reporter._output_dir

    @_reports_dir.setter
    def _reports_dir(self, output_dir):
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            res_config = self.ert().resConfig()
            base_dir = Path(res_config.model_config.getEnspath()).parent.absolute()
            self.reporter._output_dir = base_dir / output_dir
        else:
            self.reporter._output_dir = output_dir
