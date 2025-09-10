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

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity of terminal output and plotting.",
    )

    return parser


def main() -> None:
    """semeio.fmudesign is a command line utility for generating design matrices

    Wrapper for the the semeio.fmudesign module"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = get_parser()
    args = parser.parse_args()

    for sheet in ["designinput", "defaultvalues", "general_input"]:
        default = parser.get_default(sheet)
        custom = getattr(args, sheet)
        if default != custom:
            print(f"Worksheet changed from default: {default!r} -> {custom!r}")

    if not Path(args.config).is_file():
        raise OSError(f"Input file {args.config} does not exist")

    input_dict = excel2dict_design(
        args.config,
        gen_input_sheet=args.general_input,
        design_input_sheet=args.designinput,
        default_val_sheet=args.defaultvalues,
    )

    if args.config == args.destination:
        raise OSError(
            f'Identical name "{args.config}" have been provided for the input'
            "file and the output file. "
        )

    # If destination is 'analysis/generateddesignmatrix.xlsx', then plots
    # will be saved to 'analysis/generateddesignmatrix/<SENSNAME>/<VARNAME>.png'
    output_dir = Path(args.destination).parent / Path(args.destination).stem
    design = DesignMatrix(verbosity=args.verbose, output_dir=output_dir)

    design.generate(input_dict)

    design.to_xlsx(args.destination)


if __name__ == "__main__":
    main()
