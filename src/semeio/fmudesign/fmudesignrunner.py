"""Script for generating a design matrix from config input"""

import argparse
import sys
import traceback
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

from packaging.version import Version

import semeio
from semeio.fmudesign import DesignMatrix, excel_to_dict
from semeio.fmudesign.config_validation import validate_configuration


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate design matrix to be used with ERT",
        epilog=(
            "Example usage:\n"
            "  fmudesign input_config_example.xlsx \n"
            "  fmudesign input_config_example.xlsx output_example.xlsx \n\n"
            "For more information, refer to the documentation at https://equinor.github.io/fmu-tools/fmudesign.html"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )

    parser.add_argument(
        "config", type=str, help="Input design matrix filename in Excel format"
    )
    parser.add_argument(
        "destination",
        type=str,
        nargs="?",
        help="Destination filename for design matrix (default: generateddesignmatrix.xlsx)",
        default="generateddesignmatrix.xlsx",
    )
    parser.add_argument(
        "--designinput",
        type=str,
        help="Alternative sheetname for the worksheet designinput (default: designinput)",
        default="designinput",
    )
    parser.add_argument(
        "--defaultvalues",
        type=str,
        help="Alternative sheetname for worksheet defaultvalues (default: defaultvalues)",
        default="defaultvalues",
    )
    parser.add_argument(
        "--general_input",
        type=str,
        help="Alternative sheetname for the worksheet general_input (default: general_input)",
        default="general_input",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Verbosity of terminal output and plotting",
        default=0,
    )

    return parser


def validate_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    print(f"Reading file: {args.config!r}")
    for sheet in ["designinput", "defaultvalues", "general_input"]:
        default = parser.get_default(sheet)
        custom = getattr(args, sheet)
        if default != custom:
            print(f"Worksheet changed from default: {default!r} -> {custom!r}")

    if not Path(args.config).is_file():
        raise OSError(f"Input file {args.config} does not exist")
    if args.config == args.destination:
        raise OSError(
            f'Identical name "{args.config}" have been provided for the input'
            "file and the output file"
        )

    return args


def generate_design_matrix(args: Namespace) -> None:
    input_dict = excel_to_dict(
        args.config,
        gen_input_sheet=args.general_input,
        design_input_sheet=args.designinput,
        default_val_sheet=args.defaultvalues,
    )

    # Validate the input
    input_dict = validate_configuration(input_dict, input_filename=args.config)

    # If destination is 'analysis/generateddesignmatrix.xlsx', then plots
    # will be saved to 'analysis/generateddesignmatrix/<SENSNAME>/<VARNAME>.png'
    output_dir = Path(args.destination).parent / Path(args.destination).stem
    design = DesignMatrix(verbosity=args.verbose, output_dir=output_dir)

    design.generate(input_dict)
    design.to_xlsx(args.destination)


def main() -> None:
    """semeio.fmudesign is a command line utility for generating design matrices

    Wrapper for the the semeio.fmudesign module"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = get_parser()
    try:
        args = validate_args(parser)
        generate_design_matrix(args)
    except Exception:
        traceback.print_exc()
        print(
            "\n \n",
            "fmudesign failed. Read the error message above and fix the input file. \n",
            "Documentation: https://equinor.github.io/fmu-tools/fmudesign.html \n",
            "Course docs: https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html \n",
            "Issue tracker: https://github.com/equinor/semeio/issues \n",
            "If you believe this error is a bug or are unable to fix it, create an issue or contact the scout team \n",
        )
        sys.exit(1)  # Exit with a non-zero status code (required for smoke tests!)

    print(
        "\n",
        f"Thank you for using fmudesign {Version(semeio.__version__).base_version} \n",
        "Documentation: https://equinor.github.io/fmu-tools/fmudesign.html \n",
        "Course docs: https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html \n",
        "Issues/bugs/feature requests: https://github.com/equinor/semeio/issues \n",
    )


if __name__ == "__main__":
    main()
