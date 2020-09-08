import argparse
import os

from semeio.jobs.rft.utility import strip_comments


class ZoneMap:
    """A zonemap is a map from simulation grid
    k layers to a list of strings representing the
    possible zones (in a stratigraphical hiearchy of zones)
    that the k-layers belongs to.


    Args
        zones_at_k_value (dict): A dictionary from each
            k layer (as integer key larger than 0) to a list
            of strings with zone names.
    """

    def __init__(self, zones_at_k_value=None):
        self._zones_at_k_value = zones_at_k_value or {}
        self._k_values_at_zone = {}

        # Construct a reverse dictionary for the reverse
        # lookup _k_values_at_zone
        for k_value, zone_names in self._zones_at_k_value.items():
            for zone_name in zone_names:
                if zone_name not in self._k_values_at_zone:
                    self._k_values_at_zone[zone_name] = []

                self._k_values_at_zone[zone_name].append(k_value)

    @classmethod
    def load_and_parse_zonemap_file(cls, filename):
        # The job description files used in ERT does not allow
        # for optional arguments. If the user has not specified
        # a value in their configuration, the job description
        # uses the default value ZONEMAP_NOT_PROVIDED
        if filename == "ZONEMAP_NOT_PROVIDED":
            return None

        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(
                "ZoneMap file {filename} not found!".format(filename=filename)
            )

        zones_at_k_value = {}

        with open(filename, "r") as f:
            zonemap_lines = f.readlines()

        zonemap_lines = [
            (strip_comments(l), i + 1) for i, l in enumerate(zonemap_lines)
        ]
        basic_err_msg = (
            "Line {line_number} in ZoneMap file {filename} "
            "not on proper format: 'k zonename <zonename> ...'. "
        )
        for line, line_number in zonemap_lines:
            zonemap_line = line.split()

            if not zonemap_line:
                continue

            if len(zonemap_line) < 2:
                raise argparse.ArgumentTypeError(
                    basic_err_msg.format(line_number=line_number, filename=filename)
                    + "Number of zonenames must be 1 or more",
                )
            try:
                raw_k = int(zonemap_line[0])
            except ValueError as err:
                raise argparse.ArgumentTypeError(
                    basic_err_msg.format(line_number=line_number, filename=filename)
                    + "k must be integer, was {k}".format(k=zonemap_line[0])
                ) from err
            if raw_k == 0:
                raise argparse.ArgumentTypeError(
                    basic_err_msg.format(line_number=line_number, filename=filename)
                    + "k values cannot be 0, must start at 1. "
                )

            k_value = raw_k - 1
            zones = [zone.strip() for zone in zonemap_line[1:]]

            zones_at_k_value[k_value] = zones

        return cls(zones_at_k_value)

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self._zones_at_k_value
        elif isinstance(item, str):
            return item in self._k_values_at_zone
        return False

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._zones_at_k_value[item]
        elif isinstance(item, str):
            return self._k_values_at_zone[item]
        raise KeyError("{item} is neither a k value nor a zone".format(item=item))

    def has_relationship(self, zone, k):
        return k in self._k_values_at_zone.get(zone, [])
