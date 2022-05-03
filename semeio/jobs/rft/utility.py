import os
import datetime
import argparse

from ecl.rft import EclRFTFile
from ecl.grid import EclGrid


def strip_comments(line):
    return line.partition("--")[0].rstrip()


def existing_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(
            f"The path {path} is not an existing directory"
        )
    return path


def load_and_parse_well_time_file(filename):
    """
    The file must be on the format of <STRING INT INT INT INT> which
    represents <well_name day month year report_step>

    Parameters
    ----------
    filename : string
        Filename to open

    Returns
    -------
    [Tuple]
        Returns a list of tuples with (well, datetime, report_step)
    """
    # pylint: disable=too-many-locals

    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f"The path {filename} does not exist")

    well_times = []
    base_error_msg = (
        "Line {line_number} in well_and_time_file {filename} not on proper format: "
    )

    with open(filename, encoding="utf-8") as well_time_file:
        lines = well_time_file.readlines()

    well_time_lines = [(strip_comments(l), i + 1) for i, l in enumerate(lines)]

    for line, line_number in well_time_lines:
        tokens = line.split()

        if not tokens:
            continue

        if len(tokens) != 5:
            err_msg = (
                base_error_msg
                + "Unexpected number of tokens: expected 5 got {num_tokens}"
            )
            raise argparse.ArgumentTypeError(
                err_msg.format(
                    line_number=line_number, filename=filename, num_tokens=len(tokens)
                )
            )

        well = tokens[0]

        try:
            report_step = int(tokens[4])
        except ValueError as err:
            err_msg = base_error_msg + "Unable to convert {report_step} to int"
            raise argparse.ArgumentTypeError(
                err_msg.format(
                    line_number=line_number, filename=filename, report_step=tokens[4]
                )
            ) from err

        try:
            year = int(tokens[3])
            month = int(tokens[2])
            day = int(tokens[1])
            time = datetime.date(year, month, day)

        except ValueError as err:
            err_msg = base_error_msg + (
                "Unable to parse date, expected day month year got: "
                "{day} {month} {year}"
            )
            raise argparse.ArgumentTypeError(
                err_msg.format(
                    line_number=line_number,
                    filename=filename,
                    day=tokens[1],
                    month=tokens[2],
                    year=tokens[3],
                )
            ) from err

        well_times.append((well, time, report_step))

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
        ecl_rft = EclRFTFile(rft_filepath)
    except (IOError, OSError) as err:
        raise argparse.ArgumentTypeError(
            (
                f"Could not load eclipse RFT from file: {rft_filepath}\n"
                f"With the following error:"
                f"\n{err}"
            )
        )

    grid_filepath = file_path + ".EGRID"
    if not os.path.isfile(grid_filepath):
        raise argparse.ArgumentTypeError(f"The path {grid_filepath} does not exist")

    try:
        ecl_grid = EclGrid(grid_filepath)
    except (IOError, OSError) as err:
        raise argparse.ArgumentTypeError(
            (
                f"Could not load eclipse Grid from file: {grid_filepath}\n"
                f"With the following error:\n"
                f"{err}"
            )
        )

    return ecl_grid, ecl_rft
