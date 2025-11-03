"""
This module is responsible for running the 'fmudesign' CLI tool.

It contains argumenting parsing logic, some argument validation and high-level
functions that delegate to lower-level functions for creating design matrices.

There are two main sub-commands:
    $ fmudesign init  =>   Create example/demo configuration file for the user
    $ fmudesign run   =>   Run a configuration file and produce design matrix

Without arguments (init / run), the CLI will execute 'run' to be backwards
compatible. For more information, look at the code or execute

    $ fmudesign --help

"""

import argparse
import functools
import shutil
import sys
import traceback
import warnings
from argparse import ArgumentParser, Namespace, _SubParsersAction
from importlib.resources import as_file, files
from pathlib import Path

from packaging.version import Version

import semeio
from semeio.fmudesign import DesignMatrix, excel_to_dict

# These are example files that can be created with the subcommand 'fmudesign init'
EXAMPLE_FILES = {
    "fmudesign_ex_montecarlo.xlsx": "Shows all statistical parameter distributions, how to correlate samples, etc.",
    "design_input_template.xlsx": "The example file used in fmu-coursedocs",
}


def get_parser() -> tuple[ArgumentParser, _SubParsersAction]:
    """Create argument parser and return (parser, subparsers)."""

    # =============== MAIN PARSER ===============
    parser = argparse.ArgumentParser(
        description="Generates design matrices to be used with ERT",
        epilog=(
            f"""
example usage:
  $ fmudesign --help
  $ fmudesign init --help
  $ fmudesign run --help
  $ fmudesign init {next(iter(EXAMPLE_FILES.keys()))}
  $ fmudesign run {next(iter(EXAMPLE_FILES.keys()))} output_example.xlsx

getting help:
  - Documentation: https://equinor.github.io/fmu-tools/fmudesign.html
  - Issue tracker: https://github.com/equinor/semeio/issues"""
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
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # =============== SUBCOMMAND: run ===============
    description = "Generate a design matrix from a config file."
    epilog = """example usage:
  $ fmudesign --help
  $ fmudesign run input_config_example.xlsx
  $ fmudesign run input_config_example.xlsx output_example.xlsx """

    parser_run = subparsers.add_parser(
        "run",
        help=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=epilog,
        description=description,
    )

    parser_run.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser_run.add_argument(
        "config",
        type=str,
        help="Input design matrix filename in Excel format",
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
        help="Verbosity of terminal output and plotting, run with increased verbosity level -v -v to include more information",
        default=0,
    )
    func = functools.partial(subcommand_run, parser=parser_run)
    parser_run.set_defaults(func=func)

    # =============== SUBCOMMAND: init ===============
    description = "Initialize a demo file to get started with fmudesign."
    epilog = "available demo files:\n"
    ljust = max([len(f) for f in EXAMPLE_FILES])
    for filename, file_description in EXAMPLE_FILES.items():
        epilog += f"  - {filename.ljust(ljust)} : {file_description}\n"
    epilog += "\nexample usage:\n"
    epilog += f"  $ fmudesign init {next(iter(EXAMPLE_FILES.keys()))}\n"
    epilog += f"  $ fmudesign run {next(iter(EXAMPLE_FILES.keys()))}"

    parser_init = subparsers.add_parser(
        "init",
        help=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=epilog,
        description=description,
    )
    parser_init.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser_init.add_argument(
        "file", type=str, nargs="?", help="Name of demo file to create."
    )

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
    if Path(args.config).resolve() == Path(args.destination).resolve():
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


def subcommand_init(args: Namespace, parser: ArgumentParser) -> None:
    """Handles the 'init' subcommand."""
    EXAMPLES_DIR = files("semeio.fmudesign.examples")

    # Verify that all examples in EXAMPLES_DIR exist on disk
    for filename in EXAMPLE_FILES:
        assert (EXAMPLES_DIR / filename).is_file()

    # No files were provided
    if not args.file:
        parser.print_help()
        sys.exit(0)

    filename = args.file.strip()
    if filename not in EXAMPLE_FILES:
        print(f"Error on {filename!r}. Not found among: {set(EXAMPLE_FILES.keys())}")
        sys.exit(1)

    if Path(filename).exists():
        print(f"Error on {filename!r}. Already exists.")
        sys.exit(1)

    with as_file(EXAMPLES_DIR / filename) as source_path:
        shutil.copy(source_path, filename)
    print(f"Created file {filename!r}.")
    sys.exit(0)


def main() -> None:
    """semeio.fmudesign is a command line utility for generating design matrices

    Wrapper for the the semeio.fmudesign module"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser, subparsers = get_parser()

    # Backwards compatibility. If not a known command and a file, assume "run"
    if len(sys.argv) > 1:
        unknown_cmnd = sys.argv[1] not in subparsers.choices
        file_cmd = sys.argv[1].endswith((".xlsx",))
        if unknown_cmnd and file_cmd:
            sys.argv.insert(1, "run")

    args = parser.parse_args()

    # No subcommand was provided
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except Exception:
        traceback.print_exc()
        print(
            "\n \n",
            "fmudesign failed. Read the error message above and fix the input file. \n",
            " - Documentation:           https://equinor.github.io/fmu-tools/fmudesign.html \n",
            " - Course docs:             https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html \n",
            " - Issues/feature requests: https://github.com/equinor/semeio/issues \n",
            "If you believe this error is a bug or are unable to fix it, create an issue or contact the scout team \n",
        )
        sys.exit(1)  # Exit with a non-zero status code (required for smoke tests!)

    print(
        "\n",
        f"Thank you for using fmudesign {Version(semeio.__version__).base_version} \n",
        " - Documentation:           https://equinor.github.io/fmu-tools/fmudesign.html \n",
        " - Course docs:             https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html \n",
        " - Issues/feature requests: https://github.com/equinor/semeio/issues \n",
    )


if __name__ == "__main__":
    main()
