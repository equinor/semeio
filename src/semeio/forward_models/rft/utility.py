import argparse
import datetime
import os
import warnings
from pathlib import Path
from typing import List, Tuple

from resdata.grid import Grid
from resdata.rft import ResdataRFTFile


def strip_comments(line):
    return line.partition("--")[0].rstrip()


def existing_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(
            f"The path {path} is not an existing directory"
        )
    return path


def load_and_parse_well_time_file(
    filename: str,
) -> List[Tuple[str, datetime.date, int]]:
    """
    Reads and parses a file from disk, supporting 2 formats:

      <wellname : str> <date : isostring> <report_step : int>

    or the deprecated:
      <wellname : str> <day : int> <month : int> <year : int> <report_step : int>

    Returns:
        well, datetime and reportstep in a list of tuples
    """

    if not Path(filename).exists():
        raise argparse.ArgumentTypeError(f"The path {filename} does not exist")

    well_times = []
    base_error_msg = (
        "Line {line_number} in well_and_time_file {filename} not on proper format: "
    )

    lines = Path(filename).read_text(encoding="utf-8").splitlines()

    well_time_lines = [
        (strip_comments(line), idx + 1) for idx, line in enumerate(lines)
    ]

    for line, line_number in well_time_lines:
        # line_number starts at 1
        tokens = line.split()

        if not tokens:
            continue

        if len(tokens) not in {3, 5}:
            err_msg = (
                base_error_msg
                + "Unexpected number of tokens: "
                + "expected 3 or 5 (deprecated), got {num_tokens}"
            )
            raise argparse.ArgumentTypeError(
                err_msg.format(
                    line_number=line_number, filename=filename, num_tokens=len(tokens)
                )
            )

        well = tokens[0]

        report_step_token = len(tokens) - 1
        try:
            report_step = int(tokens[report_step_token])
        except ValueError as err:
            err_msg = base_error_msg + "Unable to convert {report_step} to int"
            raise argparse.ArgumentTypeError(
                err_msg.format(
                    line_number=line_number,
                    filename=filename,
                    report_step=tokens[report_step_token],
                )
            ) from err

        try:
            if len(tokens) == 3:
                welldate = datetime.datetime.fromisoformat(tokens[1]).date()
            else:
                year = int(tokens[3])
                month = int(tokens[2])
                day = int(tokens[1])
                welldate = datetime.date(year, month, day)
                warnings.warn(
                    "Use YYYY-MM-DD as date format for gendata_rft input",
                    FutureWarning,
                    stacklevel=1,
                )

        except ValueError as err:
            err_msg = base_error_msg + (f"Unable to parse date from the line: {line}")
            raise argparse.ArgumentTypeError(
                err_msg.format(
                    line_number=line_number,
                    filename=filename,
                    line=line,
                )
            ) from err

        well_times.append((well, welldate, report_step))

    return well_times


def valid_eclbase(file_path):
    """
    The filename is assumed to be without extension and two files
    must be present, <filename>.RFT and <filename>.EGRID.
    Loads both files with respective loaders and returns them

    Parameters
    ----------
    filename : string
        Filename to open

    Returns
    -------
    Tuple
        Returns a tuple with an ecl grid instance and an rft instance
    """
    rft_filepath = file_path + ".RFT"
    if not os.path.isfile(rft_filepath):
        raise argparse.ArgumentTypeError(f"The path {rft_filepath} does not exist")

    try:
        ecl_rft = ResdataRFTFile(rft_filepath)
    except OSError as err:
        raise argparse.ArgumentTypeError(
            f"Could not load eclipse RFT from file: {rft_filepath}\n"
            f"With the following error:"
            f"\n{err}"
        ) from err

    grid_filepath = file_path + ".EGRID"
    if not os.path.isfile(grid_filepath):
        raise argparse.ArgumentTypeError(f"The path {grid_filepath} does not exist")

    try:
        ecl_grid = Grid(grid_filepath)
    except OSError as err:
        raise argparse.ArgumentTypeError(
            f"Could not load eclipse Grid from file: {grid_filepath}\n"
            f"With the following error:\n"
            f"{err}"
        ) from err

    return ecl_grid, ecl_rft
