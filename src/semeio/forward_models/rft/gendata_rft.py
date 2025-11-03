import datetime
import logging
import os
from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd
from resdata.grid import Grid
from resdata.rft import ResdataRFTFile

from semeio.forward_models.rft.trajectory import Trajectory
from semeio.forward_models.rft.zonemap import ZoneMap

logger = logging.getLogger(__name__)


def _write_gen_data_files(
    trajectory_df: pd.DataFrame, output_directory: str, well_name: str, report_step: int
) -> None:
    """Generate three files with the information GEN_DATA needs
    from the trajectory dataframe.

    See https://github.com/equinor/ert/blob/0dc96c49ca8eafee54a06227530b5a47b094e2bc/docs/tips.txt#L87

    One main file (RFT_{well_name}_{report_step}) will always be produced with
    pressure values, and two auxiliary files will be produced with the suffixes
    "_active" and "_inactive_info". The _active file tells ERT if some of the
    values in the base file should be ignored, and _inactive_info will give an
    explanation for why it should be ignored.

    The pressure file will always be written. If there is no pressure
    data, a file with -1 values will be written for each point pr. line.

    If there is saturation data in the dataframe (in the columns swat,
    sgas and/or soil), these will be written to separate files, with e.g. the
    string SWAT injected in the emitted filename.

    Args:
        trajectory_df: The column "order" should contain
            the original row order from input text files. "pressure" should
            hold pressure data, but will be defaulted if not present. "is_active"
            should be a boolean column, and "inactive_info" should contain
            strings with information on why points are inactive.
        output_directory: Directory name, for where to dump files.
        well: Name of well, to be used to construct filenames.
        report_step: The RFT report step, used to construct filenames
    """
    data2fname = {"pressure": "", "swat": "SWAT_", "sgas": "SGAS_", "soil": "SOIL_"}
    for dataname in {"pressure"}.union(
        set(trajectory_df.columns).intersection(data2fname)
    ):
        _write_simdata(
            os.path.join(
                output_directory,
                f"RFT_{data2fname[dataname]}{well_name}_{report_step}",
            ),
            dataname,
            trajectory_df,
        )
    _write_active(
        os.path.join(output_directory, f"RFT_{well_name}_{report_step}") + "_active",
        trajectory_df,
    )
    _write_inactive_info(
        os.path.join(output_directory, f"RFT_{well_name}_{report_step}")
        + "_inactive_info",
        trajectory_df,
    )


def _write_simdata(
    output_file: str, dataname: str, trajectory_df: pd.DataFrame
) -> None:
    """Write pressure value, one per line for all points, -1 is used where
    there is no pressure information.
    """
    with open(output_file, "w+", encoding="utf-8") as file_handle:
        if dataname in trajectory_df:
            file_handle.write(
                "\n".join(
                    trajectory_df.sort_values("order")[dataname]
                    .fillna(value=-1)
                    .astype(str)
                    .values
                )
                + "\n"
            )
        else:
            file_handle.write("\n".join(["-1"] * len(trajectory_df)) + "\n")

    logger.info(f"Forward model script gendata_rft.py: Wrote file {output_file}")


def _write_active(output_file: str, trajectory_df: pd.DataFrame) -> None:
    """Write a file with "1" pr row if a point is active, "0" if not"""
    with open(output_file, "w+", encoding="utf-8") as file_handle:
        file_handle.write(
            "\n".join(
                trajectory_df.sort_values("order")["is_active"]
                .astype(int)
                .astype(str)
                .values
            )
        )


def _write_inactive_info(output_file: str, trajectory_df: pd.DataFrame) -> None:
    """Write a file with explanations to users for inactive points"""
    with open(output_file, "w+", encoding="utf-8") as file_handle:
        if "inactive_info" not in trajectory_df:
            file_handle.write("")
        else:
            file_handle.write(
                "\n".join(
                    trajectory_df[~trajectory_df["is_active"]]
                    .sort_values("order")["inactive_info"]
                    .dropna()
                    .values
                )
            )


def _populate_trajectory_points(
    well: str,
    date: datetime.date,
    trajectory_points: Trajectory,
    ecl_grid: Grid,
    ecl_rft: ResdataRFTFile,
    zonemap: ZoneMap | None = None,
) -> Trajectory | None:
    """
    Populate a list of trajectory points, that only contain UTM coordinates
    for a well-path, with (i,j,k) indices corresponding to a given Eclipse grid,
    and simulated pressure at the given date in the cells.

    Args:
        well: Eclipse-name of well
        date: Date for which the RFT observation should be taken,
            must correspond to a date in the Eclipse RFT binary output
        trajectory_points: Representing the wellpath in UTM at which
            the real life RFT observations points are valid.
        ecl_grid: The grid where the (i,j,k) are looked up
        ecl_rft: THe rft file where pressure for the point is found
        zonemap: The mapping of k indices to zones, used to validate
            that the found ijk-coordinates is in the correct zone.

    Returns:
        The list of trajectory points, augmented with i,j,k, simulated pressure
        and the zone name if zonemap is provided.
    """
    try:
        rft = ecl_rft.get(well, date)
    except KeyError:
        logger.error(
            "Forward model script gendata_rft.py: "
            f"No RFT data found for well {well} at date {date}"
        )
        return None

    ijk_guess = None

    start_time = perf_counter()
    for point in iter(trajectory_points):
        ijk = ecl_grid.find_cell(
            point.utm_x, point.utm_y, point.true_vertical_depth, start_ijk=ijk_guess
        )
        point.set_ijk(ijk)
        ijk_guess = ijk
        point.update_simdata_from_rft(rft)
        point.validate_zone(zonemap)

    logger.info(f"Finding points took {perf_counter() - start_time}s")

    try:
        from resfo_utilities import CornerpointGrid  # noqa: PLC0415

        grid = CornerpointGrid.read_egrid(ecl_grid.get_name())
        start_time = perf_counter()
        new_points = grid.find_cell_containing_point(
            np.array(
                [
                    (point.utm_x, point.utm_y, point.true_vertical_depth)
                    for point in iter(trajectory_points)
                ],
                dtype=np.float32,
            )
        )
        logger.info(
            f"Finding points with new method took {perf_counter() - start_time}s"
        )

        for point, new_point in zip(iter(trajectory_points), new_points, strict=True):
            if point.grid_ijk != new_point:
                logger.warning(
                    "New and old cell finding method"
                    " did not agree on grid coordinates: "
                    f"{point.grid_ijk} vs {new_point}"
                )

    except Exception as err:
        logger.error(f"Got error {err} while reading egrid with resfo")

    return trajectory_points


def run(
    well_times: Sequence[tuple[str, datetime.date, int]],
    trajectories: Mapping[str, Trajectory],
    ecl_grid: Grid,
    ecl_rft: ResdataRFTFile,
    zonemap: ZoneMap | None = None,
    csvfile: str | None = None,
    outputdirectory: str = ".",
) -> None:
    dframes: list[pd.DataFrame] = []

    if not well_times:
        raise ValueError("No RFT data requested")

    for well, time, report_step in well_times:
        logger.debug(
            f"Collecting RFT for well: {well} at date: {time}, "
            f"report step: {report_step}"
        )

        trajectory_points = _populate_trajectory_points(
            well=well,
            date=time,
            trajectory_points=trajectories[well],
            ecl_grid=ecl_grid,
            ecl_rft=ecl_rft,
            zonemap=zonemap,
        )

        if trajectory_points:
            # Aggregate the same data to a dataframe,
            # each trajectory tagged by well and time:
            trajectory_df = trajectory_points.to_dataframe(
                zonemap=zonemap,
            ).assign(
                well=well,
                time=time,
                report_step=report_step,
            )

            # Write trajectory and associated data to files, one file pr.
            # datatype, well and time, this is for GEN_DATA to pick up.
            _write_gen_data_files(trajectory_df, outputdirectory, well, report_step)

            # Aggregate dataframe for all wells and report steps.
            dframes.append(trajectory_df)
        else:
            logger.error(f"No trajectory points for well {well} at date: {time} found")

    if csvfile is not None and dframes:
        pd.concat(dframes, ignore_index=True, sort=False).to_csv(csvfile, index=False)

    if len(dframes) < len(well_times):
        raise ValueError("Failed to extract all requested RFT data")
