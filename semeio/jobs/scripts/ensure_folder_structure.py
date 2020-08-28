#!/usr/bin/env python
import argparse
import yaml

from semeio.jobs import ensure_folder_structure


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            """Ensures that the specified folder structure is present on disk and if not,
        creates it.

        Example:
        a:
          b:
          c:
            d:
        e:
        f:

        Would result in three folders (a, e and f) existing in the root folder.
        Two folders (b and c) will exist inside a. And inside c again, a single
        folder named d will reside.
        """
        )
    )
    parser.add_argument(
        "configfile", help="The folder structure configuration",
    )
    parser.add_argument(
        "-r", "--root", dest="root", default=".",
    )
    return parser.parse_args()


def _load_config(configfile):
    with open(configfile) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = _parse_arguments()
    folder_structure = _load_config(args.configfile)
    ensure_folder_structure.run(folder_structure, args.root)
