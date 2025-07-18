"""Script for generating a design matrix from config input"""

import argparse
import warnings
from argparse import ArgumentParser
from pathlib import Path

from semeio.fmudesign import DesignMatrix, excel2dict_design


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate design matrix to be used with ert DESIGN2PARAMS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config", type=str, help="Input design config filename in Excel format"
    )
    parser.add_argument(
        "destination",
        type=str,
        nargs="?",
        help="Destination filename for design matrix",
        default="generateddesignmatrix.xlsx",
    )
    parser.add_argument(
        "--designinput",
        type=str,
        help="Alternative sheetname for the worksheet designinput",
        default="designinput",
    )
    parser.add_argument(
        "--defaultvalues",
        type=str,
        help="Alternative sheetname for worksheet defaultvalues",
        default="defaultvalues",
    )
    parser.add_argument(
        "--general_input",
        type=str,
        help="Alternative sheetname for the worksheet general_input",
        default="general_input",
    )

    return parser


def main() -> None:
    """semeio.fmudesign is a command line utility for generating design matrices

    Wrapper for the the semeio.fmudesign module"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = get_parser()
    args = parser.parse_args()

    # Defaulted options should be reset to None, so that the other
    # defaulting level inside _excel2dict can do its work.
    if args.designinput == parser.get_default("designinput"):
        args.designinput = None
    if args.defaultvalues == parser.get_default("defaultvalues"):
        args.defaultvalues = None
    if args.general_input == parser.get_default("general_input"):
        args.general_input = None

    sheetnames = {}
    if args.designinput:
        sheetnames["designinput"] = args.designinput
    if args.defaultvalues:
        sheetnames["defaultvalues"] = args.defaultvalues
    if args.general_input:
        sheetnames["general_input"] = args.general_input

    if sheetnames:
        print("Worksheets changed from default:")
        print(sheetnames)

    if isinstance(args.config, str):
        if not Path(args.config).is_file():
            raise OSError(f"Input file {args.config} does not exist")
        input_dict = excel2dict_design(args.config, sheetnames)

    if args.config == args.destination:
        raise OSError(
            f'Identical name "{args.config}" have been provided for the input'
            "file and the output file. "
            " Exiting....."
        )

    design = DesignMatrix()

    design.generate(input_dict)

    Path(args.destination).parent.mkdir(exist_ok=True, parents=True)

    design.to_xlsx(args.destination)


if __name__ == "__main__":
    main()
