import argparse
import logging

from semeio import valid_file
from semeio._exceptions.exceptions import ValidationError
from semeio.forward_models.design2params import design2params

description = """
Reads a design matrix in XLSX-format and

* Fetches values belonging to given ERT realization and *appends*
  key-value pairs to parameters.txt
* Converts specified worksheet to txt-format and puts designmatrix.txt in RUNPATH
* Creates designparameters.txt which only contains variables taken from the design
  matrix

Requires a matrix with column header as strings in topmost row in xls-file.
Row 2 in xls-file must then correspond with the data you want for realization-0
Column 1 in xls-file should contain the realization number, and will be ignored
by this script.  You must run this script as a FORWARD_MODEL from your ERT config,
before you run DESIGN_KW.
"""


def create_parser():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "realization",
        type=int,
        help="Which realization",
    )

    parser.add_argument(
        "xlsfilename",
        type=valid_file,
        help="Path to design-matrix (xlsx file)",
    )

    parser.add_argument(
        "designsheetname",
        type=str,
        help="Design sheet name",
    )

    parser.add_argument(
        "defaultssheetname",
        type=str,
        nargs="?",
        help="Defaults sheet name",
        default=None,
    )

    parser.add_argument(
        "--parametersfilename",
        "-p",
        required=False,
        default=design2params._PARAMETERS_TXT,
        type=str,
    )

    parser.add_argument(
        "--log-level",
        "-l",
        required=False,
        default="WARNING",
        type=logging.getLevelName,
    )

    return parser


def main_entry_point():
    parser = create_parser()
    parsed_args = parser.parse_args()
    try:
        design2params.run(
            realization=parsed_args.realization,
            xlsfilename=parsed_args.xlsfilename,
            designsheetname=parsed_args.designsheetname,
            defaultssheetname=parsed_args.defaultssheetname,
            parametersfilename=parsed_args.parametersfilename,
            log_level=parsed_args.log_level,
        )
    except ValidationError as err:
        raise SystemExit(str(err)) from err
