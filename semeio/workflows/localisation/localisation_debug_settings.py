from enum import IntEnum


class LogLevel(IntEnum):
    OFF = 0
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3
    LEVEL4 = 4


# Global variables initialized with default
debug_level = LogLevel.OFF
scaling_param_number = 1


def debug_print(text, threshold=LogLevel.OFF):
    if debug_level > LogLevel.OFF:
        if threshold == LogLevel.LEVEL1:
            text = "-- " + text
        elif threshold == LogLevel.LEVEL2:
            text = "  -- " + text
        elif threshold == LogLevel.LEVEL3:
            text = "    -- " + text
        if debug_level >= threshold:
            print(text)
