import logging
from enum import IntEnum


class LogLevel(IntEnum):
    OFF = 0
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3
    LEVEL4 = 4


def debug_print(text, threshold, log_level=LogLevel.OFF):
    if threshold <= log_level:
        # Use the logging module to log the info to file
        logging.info("  " * threshold + "-- " + text)
