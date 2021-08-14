from enum import IntEnum


class LogLevel(IntEnum):
    OFF = 0
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3
    LEVEL4 = 4


class LocalDebugLog:
    # pylint: disable=R0903
    level = LogLevel.LEVEL1

    @classmethod
    def debug_print(cls, text, threshold=LogLevel.OFF):
        if cls.level > LogLevel.OFF:
            if threshold == LogLevel.LEVEL1:
                text = "-- " + text
            elif threshold == LogLevel.LEVEL2:
                text = "  -- " + text
            elif threshold == LogLevel.LEVEL3:
                text = "    -- " + text
            if cls.level >= threshold:
                print(text)


def debug_print(text, threshold=LogLevel.OFF):
    LocalDebugLog.debug_print(text, threshold)
