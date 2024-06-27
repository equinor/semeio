import datetime
import logging
import os
import threading
from logging.handlers import BufferingHandler
from pathlib import Path
from types import MethodType

from ert import ErtScript, LibresFacade

from semeio.communication.reporter import FileReporter

SEMEIOSCRIPT_LOG_FILE = "workflow-log.txt"


class _LogHandlerContext:
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
            self._reporter.publish_msg(self._namespace, _format_record(log_record))

        super().flush()


def _format_record(log_record):
    return (
        f"{log_record.levelname} "
        f"[{datetime.datetime.fromtimestamp(log_record.created)}]: "
        f"{log_record.message}"
    )


class SemeioScript(ErtScript):
    """
    SemeioScript is a workflow utility extending the functionality of ErtScript.
    In particular it provides a `self.reporter` instance available for passing
    data to the common storage. In addition, while `self.run` is being executed
    it forwards log statements to the reporter as well.
    """

    def __init__(self):
        super().__init__()
        self.facade = None
        self._output_dir = None
        self._reporter = None
        self._wrap_run()

    def _wrap_run(self):
        self._real_run = self.run

        def run_with_handler(self, *args, **kwargs):
            log = logging.getLogger("")
            thread_id = threading.get_ident()
            self.facade = LibresFacade(self.ert())
            self._output_dir = self._get_output_dir()
            report_handler = _ReportHandler(self._output_dir, thread_id)
            with _LogHandlerContext(log, report_handler):
                self._reporter = FileReporter(self._output_dir)
                return self._real_run(*args, **kwargs)

        self.run = MethodType(run_with_handler, self)

    def _get_output_dir(self):
        base_dir = Path(self.facade.enspath).parent.absolute()
        try:
            sub_dir = self.ensemble.name
        except AttributeError:
            sub_dir = "default"
        return os.path.join(
            base_dir,
            "reports",
            Path(self.facade.user_config_file).stem,
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
            base_dir = Path(self.facade.enspath).parent.absolute()
            self.reporter._output_dir = base_dir / output_dir
        else:
            self.reporter._output_dir = output_dir
