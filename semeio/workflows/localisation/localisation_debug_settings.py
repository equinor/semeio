from enum import IntEnum


class LogLevel(IntEnum):
    OFF = 0
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3
    LEVEL4 = 4


# Global variables initialized with default
debug_level = LogLevel.OFF
scaling_parameter_number = 1


def debug_print(text, level=LogLevel.OFF):
    if debug_level > LogLevel.OFF:
        if level == LogLevel.LEVEL1:
            text = "-- " + text
        elif level == LogLevel.LEVEL2:
            text = "  -- " + text
        elif level == LogLevel.LEVEL3:
            text = "    -- " + text
        if debug_level >= level:
            print(text)
