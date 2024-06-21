import argparse
import logging
import os
import sys

from semeio.forward_models.rft import gendata_rft
from semeio.forward_models.rft.trajectory import Trajectory
from semeio.forward_models.rft.utility import (
    existing_directory,
    load_and_parse_well_time_file,
    valid_eclbase,
)
from semeio.forward_models.rft.zonemap import ZoneMap

logger = logging.getLogger(__name__)

description = """
The GENDATA_RFT forward model is used to extract RFT data (pressure and
saturation) for specific points in the reservoir (typically along well
trajectories) from an Eclipse output file (or equivalent data format), given a file
containing relevant well names and their corresponding dates. It is also
possible to specify a zonemap that validates each zone specified in the
trajectory files.

Pressure data is outputted to files like ``RFT_A-1_0`` to a selected
directory, this would correspond to well A-1 at report step 0. Saturation data,
if available, would be exported to ``RFT_A-1_SWAT_0``, and similarly for SGAS
and SOIL.

The user will be given error messages and the applications stops if there is no RFT
available for the well(s) at the date(s).

A CSV file is prepared with all extracted data. This can be further used for
visualization in Webviz. In order to merge this CSV with observed values, see
``merge_rft_ertobs`` from subscript.

The pressure or saturation data is set to -1 and the inactive value to 0 if any
of the following applies:

* The trajectory point can not be mapped to a cell in the grid
* There is no RFT data for the cell
* The zone mapping is invalid.

For setting up the input to this forward model, the tool ``create_rft_ertobs``
from fmu.tools can be used, called from within RMS. See
https://equinor.github.io/fmu-tools/create_rft_ertobs.html
"""


def _build_parser():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-e",
        "--eclbase",
        type=valid_eclbase,
        required=True,
        help="""
Filepath to an eclipse base (or with equivalent data format) that
contains both RFT and EGRID files. Note that the proper extensions are added,
so it is only necessary to specify the basename
""",
    )
    parser.add_argument(
        "-w",
        "--well_and_time_file",
        type=load_and_parse_well_time_file,
        required=True,
        help="""
Filepath to a file containing the well and time that will be loaded from the RFT
file. The accepted file format is: <well_name> <YYYY-MM-DD> <report>.
For each well name given in the file there must also exist a <well_name.txt> file
that contains a list of trajectory points. Each line must be specified according
to: <utmx> <utmy> <TVD> <MD>
""",
    )
    parser.add_argument(
        "-t",
        "--trajectory_path",
        required=True,
        type=existing_directory,
        help="""
Path to the directory that contains the trajectory files referenced from the well
and time file. Each well must have a trajectory file that contains at least one
trajectory point.
""",
    )
    parser.add_argument(
        "-z",
        "--zonemap",
        required=False,
        type=ZoneMap.load_and_parse_zonemap_file,
        help="""
Path to a file with zonemap information. This file must include every k-value that
any RFT value is located within, irregardless if the trajectory point contains a
zone definition. If a k value does not exist the program will stop. The zone
mapping will be validated and if it is invalid, the pressure is set to -1 and the
inactive value to 0
""",
    )
    parser.add_argument(
        "-c",
        "--csvfile",
        required=False,
        type=str,
        default="gendata_rft.csv",
        help="""
Path to a file to which a CSV output is dumped. The CSV will contain all the
information of the wellspaths, times, pressures and zone information, and whether
a specific pressure point is valid
""",
    )
    parser.add_argument(
        "-o",
        "--outputdirectory",
        required=False,
        type=existing_directory,
        default=".",
        help=(
            "Output directory used for outputting resulting files "
            "(except for the csvfile output). "
            "The directory must exist. "
            "Defaults to RUNPATH."
        ),
    )
    parser.add_argument(
        "--log-level",
        "-l",
        required=False,
        default="WARNING",
        type=logging.getLevelName,
        help="Sets the log level",
    )

    return parser


def main_entry_point():
    arg_parser = _build_parser()
    options = arg_parser.parse_args()
    logger.setLevel(options.log_level)

    context_errors = []
    trajectories = {}
    for well_name, *_ in options.well_and_time_file:
        try:
            trajectories[well_name] = Trajectory.load_from_file(
                filepath=os.path.join(options.trajectory_path, well_name + ".txt")
            )
        except (OSError, ValueError) as err:
            context_errors.append(str(err))

    if context_errors:
        raise SystemExit("\n".join(context_errors))

    logger.info("All files loaded\nRetrieving RFT data...")

    try:
        gendata_rft.run(
            well_times=options.well_and_time_file,
            trajectories=trajectories,
            ecl_grid=options.eclbase[0],
            ecl_rft=options.eclbase[1],
            zonemap=options.zonemap,
            csvfile=options.csvfile,
            outputdirectory=options.outputdirectory,
        )
        with open("GENDATA_RFT.OK", "w", encoding="utf-8") as file_handle:
            file_handle.write("GENDATA RFT completed OK")
        logger.info("Completed!")
    except ValueError as exception:
        logger.error(str(exception))
        sys.exit(1)
