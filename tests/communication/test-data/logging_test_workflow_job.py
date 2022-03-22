import logging
from threading import Thread

from semeio.communication import SemeioScript


class TestWorkflowJob(SemeioScript):
    def run(self, *args):
        self.reporter.publish("test_data", list(range(10)))

        # The mission of this code is to simulate that something outside the
        # thread of the workflow (e.g. ERT itself), is logging. This should not
        # end up in the workflow log itself.
        thread = Thread(
            target=lambda: logging.error(
                "Log statement from outside the workflow thread."
            )
        )
        thread.start()
        thread.join()

        logging.error(
            "I finished without any problems - hence I'm not a failure after all!"
        )
