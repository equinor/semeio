import re

from semeio.reporter.serialize_numpy import serialize_numpy
from semeio.reporter.serialize_json import serialize_json


class Reporter(object):
    _serializers = [serialize_numpy, serialize_json]
    _valid_key = re.compile("^[0-9a-zA-Z-_+.]+$")

    def __init__(self):
        pass

    def report(self, key, val):
        if not isinstance(key, str):
            raise TypeError("Report key must be of type str")
        if key.match(self._valid_key):
            raise ValueError("Report key {} malformed: May only contain alphanumerics, '_', '-', '+', '.'", repr(key))

        for ser in self._serializers:
            if ser(key, val):
                return
        raise TypeError(
            "Reporter cannot serialize values of type '{}'".format(type(val).__name__)
        )


_reporter = Reporter()


def report(key, val):
    _reporter.report(key, val)
