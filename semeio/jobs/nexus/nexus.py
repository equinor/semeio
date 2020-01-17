import logging
import sys


import logging


logger = logging.getLogger(__name__)


def run(
    nexusfile,
    eclbase,
    refcase,
    history,
    log_level
):
    """
    Reads out all file content from different files and create dataframes
    """
    logger.setLevel(log_level)

   