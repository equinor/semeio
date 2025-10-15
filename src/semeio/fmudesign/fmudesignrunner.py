"""Script for generating a design matrix from config input"""

import argparse
import functools
import shutil
import sys
import traceback
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

from packaging.version import Version

import semeio
from semeio.fmudesign import DesignMatrix, excel_to_dict


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
        "-v",
        "--version",
        action="version",
        version=f"fmudesign {Version(semeio.__version__)}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =============== SUBCOMMAND: run ===============
    parser_run = subparsers.add_parser("run", help="Run fmudesign on an input file")
    # parser_run.add_argument(
    #    "-h", "--help", action="help", help="Show this help message and exit"
    # )
    parser_run.add_argument(
        "config", type=str, help="Input design matrix filename in Excel format"
    )
    parser_run.add_argument(
        "destination",
        type=str,
        nargs="?",
        help="Destination filename for design matrix (default: generateddesignmatrix.xlsx)",
        default="generateddesignmatrix.xlsx",
    )
    parser_run.add_argument(
        "--designinput",
        type=str,
        help="Alternative sheetname for the worksheet designinput (default: designinput)",
        default="designinput",
    )
    parser_run.add_argument(
        "--defaultvalues",
        type=str,
        help="Alternative sheetname for worksheet defaultvalues (default: defaultvalues)",
        default="defaultvalues",
    )
    parser_run.add_argument(
        "--general_input",
        type=str,
        help="Alternative sheetname for the worksheet general_input (default: general_input)",
        default="general_input",
    )
    parser_run.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Verbosity of terminal output and plotting",
        default=0,
    )
    func = functools.partial(subcommand_run, parser=parser_run)
    parser_run.set_defaults(func=func)

    # =============== SUBCOMMAND: init ===============
    parser_init = subparsers.add_parser(
        "init", help="Create an input file (learning/demo)"
    )
    parser_init.add_argument("files", nargs="*", help="Files to create")

    func = functools.partial(subcommand_init, parser=parser_init)
    parser_init.set_defaults(func=func)

    return parser, subparsers


def subcommand_run(args: Namespace, parser: ArgumentParser) -> None:
    """Handles the 'run' subcommand."""

    # Check if defaults were changed
    for sheet in ["designinput", "defaultvalues", "general_input"]:
        default = parser.get_default(sheet)
        custom = getattr(args, sheet)
        if default != custom:
            print(f"Worksheet changed from default: {default!r} -> {custom!r}")

    # Check existence of config file
    if not Path(args.config).is_file():
        raise OSError(f"Input file {args.config} does not exist")

    # Check if destination exists
    if args.config == args.destination:
        raise OSError(
            f'Identical name "{args.config}" have been provided for the input'
            "file and the output file"
        )

    # Parse Excel config file to dict-of-dict configuration
    print(f"Reading file: {args.config!r}")
    config = excel_to_dict(
        args.config,
        gen_input_sheet=args.general_input,
        design_input_sheet=args.designinput,
        default_val_sheet=args.defaultvalues,
    )

    # If destination is 'analysis/generateddesignmatrix.xlsx', then plots
    # will be saved to 'analysis/generateddesignmatrix/<SENSNAME>/<VARNAME>.png'
    output_dir = Path(args.destination).parent / Path(args.destination).stem
    design = DesignMatrix(verbosity=args.verbose, output_dir=output_dir)

    design.generate(config)
    design.to_xlsx(args.destination)


def create_example_file(filename: str) -> None:
    """Copies a file 'filename' (e.g. 'example.xlsx') from the examples
    directory and creates it in the current directory."""

    examples_dir = Path(__file__).parent / "examples"

    if not (examples_dir / filename).exists():
        raise OSError(f"Input file {filename!r} does not exist")

    shutil.copy(examples_dir / filename, filename)


def subcommand_init(args: Namespace, parser: ArgumentParser) -> None:
    """Handles the 'init' subcommand."""
    EXAMPLES_dIR = Path(__file__).parent / "examples"
    FILES = {
        "example1.xlsx": "The first example file you should run!",
        "example2.xlsx": "The second example file you should run!",
        "simple.yaml": "simple be good",
    }

    for filename in FILES:
        assert (EXAMPLES_dIR / filename).exists()

    # The user did not specify any demo files
    if not args.files:
        print("Available demo input files:")
        for filename, file_description in FILES.items():
            print(f" - {filename!r} : {file_description}")
        print("To create one or more of these files, use commands like:")
        print(f"$ fmudesign init {next(iter(FILES.keys()))}")
        sys.exit(0)

    # The user did specify one or more demo files
    for filename in args.files:
        if filename not in FILES:
            print(f"Skipping file {filename!r}. Not found among: {set(FILES.keys())}")

        if Path(filename).exists():
            print(f"Skipping file {filename!r}. Already exists.")

        shutil.copy(EXAMPLES_dIR / filename, filename)
        print(f"Created file {filename!r}.")


def main() -> None:
    """semeio.fmudesign is a command line utility for generating design matrices

    Wrapper for the the semeio.fmudesign module"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser, subparsers = get_parser()

    # Backwards compatibility. If not a known command and a file, assume "run"
    if (sys.argv[1] not in subparsers.choices) and (sys.argv[1].endswith((".xlsx",))):
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    try:
        args.func(args)
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
