from .serialize_numpy import serialize_numpy
from .serialize_json import serialize_json


class Reporter(object):
    _serializers = [serialize_numpy, serialize_json]

    def __init__(self):
        pass

    def report(self, key, val):
        if not isinstance(key, str):
            raise TypeError("Report key must be of type str")
        if key == "" or "/" in key:
            raise ValueError("Invalid report key")

        for ser in self._serializers:
            if ser(key, val):
                return
        raise TypeError(
            "Reporter cannot serialize values of type '{}'".format(type(val).__name__)
        )


_reporter = Reporter()


def report(key, val):
    _reporter.report(key, val)
