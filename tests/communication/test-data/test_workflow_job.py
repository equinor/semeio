import logging
from semeio.communication import SemeioScript


class TestWorkflowJob(SemeioScript):
    def run(self, *args):
        self.reporter.publish("test_data", list(range(10)))
        logging.error(
            "I finished without any problems - hence I'm not a failure after all!"
        )
