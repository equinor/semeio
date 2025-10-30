import argparse
import os
from collections.abc import Container, Mapping
from typing import Any, Self

from semeio.forward_models.rft.utility import strip_comments


class ZoneMap:
    """A zonemap is a map from simulation grid
    k layers to a list of strings representing the
    possible zones (in a stratigraphical hierarchy of zones)
    that the k-layers belongs to.


    Args:
        zones_at_k_value: A mapping from each
            k layer (as integer key larger than 0) to a list
            of strings with zone names.
    """

    def __init__(
        self, zones_at_k_value: Mapping[int, Container[str]] | None = None
    ) -> None:
        self._zones_at_k_value = zones_at_k_value or {}

    @classmethod
    def load_and_parse_zonemap_file(cls, filename: str) -> Self | None:
        # The forward model description files used in ERT does not allow
        # for optional arguments. If the user has not specified
        # a value in their configuration, the forward model description
        # uses the default value ZONEMAP_NOT_PROVIDED
        if filename == "ZONEMAP_NOT_PROVIDED":
            return None

        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f"ZoneMap file {filename} not found!")

        zones_at_k_value: dict[int, Any] = {}

        with open(filename, encoding="utf-8") as file_handle:
            _zonemap_lines = file_handle.readlines()

        zonemap_lines = [
            (strip_comments(line), idx + 1) for idx, line in enumerate(_zonemap_lines)
        ]
        base_err_msg = (
            "Line {line_number} in ZoneMap file {filename} "
            "not on proper format: 'k zonename <zonename> ...'. "
        )
        for line, line_number in zonemap_lines:
            zonemap_line = line.split()

            if not zonemap_line:
                continue

            if len(zonemap_line) < 2:
                raise argparse.ArgumentTypeError(
                    base_err_msg.format(line_number=line_number, filename=filename)
                    + "Number of zonenames must be 1 or more",
                )
            try:
                raw_k = int(zonemap_line[0])
            except ValueError as err:
                raise argparse.ArgumentTypeError(
                    base_err_msg.format(line_number=line_number, filename=filename)
                    + f"k must be integer, was {zonemap_line[0]}"
                ) from err
            if raw_k == 0:
                raise argparse.ArgumentTypeError(
                    base_err_msg.format(line_number=line_number, filename=filename)
                    + "k values cannot be 0, must start at 1. "
                )

            k_value = raw_k - 1
            zones = [zone.strip() for zone in zonemap_line[1:]]

            zones_at_k_value[k_value] = zones

        return cls(zones_at_k_value)

    def __contains__(self, item: int) -> bool:
        return item in self._zones_at_k_value

    def __getitem__(self, item: int) -> Container[str]:
        return self._zones_at_k_value[item]

    def has_relationship(self, zone: str, k: int) -> bool:
        return zone in self._zones_at_k_value.get(k, [])
