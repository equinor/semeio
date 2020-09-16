import os

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _write_gen_data_files(trajectory_df, fname):
    """Generate three files with the information GEN_DATA needs
    from the trajectory dataframe.

    See https://github.com/equinor/ert/blob/0dc96c49ca8eafee54a06227530b5a47b094e2bc/docs/tips.txt#L87

    One main file (fname) will be produced with pressure values, and two
    auxiliary files will be produced with the suffixes "_active" and
    "_inactive_info". The _active file tells ERT if some of the values
    in the base file should be ignored, and _inactive_info will give
    an explanation for why it should be ignored.

    Args:
        trajectory_df (pd.DataFrame): The column "order" should contain
            the original row order from input text files. "pressure" should
            hold pressure data, but will be defaulted if not present. "is_active"
            should be a boolean column, and "inactive_info" should contain
            strings with information on why points are inactive.
        fname (str): Filename to hold the values for the GEN_DATA keyword,
            which are pressure values in this case, and used as a basename
            for the two auxiliary files.
    """  # noqa
    _write_pressure(fname, trajectory_df)
    _write_active(fname + "_active", trajectory_df)
    _write_inactive_info(fname + "_inactive_info", trajectory_df)


def _write_pressure(fname, trajectory_df):
    """Write pressure value, one pr line for all points, -1 is used where
    there is no pressure information.
    """
    with open(fname + "", "w+") as fh:
        if "pressure" in trajectory_df:
            fh.write(
                "\n".join(
                    trajectory_df.sort_values("order")["pressure"]
                    .fillna(value=-1)
                    .astype(str)
                    .values
                )
                + "\n"
            )
        else:
            fh.write("\n".join(["-1" * len(trajectory_df)]))

    logger.info("Forward model script gendata_rft.py: Wrote file {}".format(fname))


def _write_active(fname, trajectory_df):
    """Write a file with "1" pr row if a point is active, "0" if not"""
    with open(fname, "w+") as fh:
        fh.write(
            "\n".join(
                trajectory_df.sort_values("order")["is_active"]
                .astype(int)
                .astype(str)
                .values
            )
        )


def _write_inactive_info(fname, trajectory_df):
    """Write a file with explanations to users for inactive points"""
    with open(fname, "w+") as fh:
        if "inactive_info" not in trajectory_df:
            fh.write("")
        else:
            fh.write(
                "\n".join(
                    trajectory_df[~trajectory_df["is_active"]]
                    .sort_values("order")["inactive_info"]
                    .dropna()
                    .values
                )
            )


def _populate_trajectory_points(
    well, date, trajectory_points, ecl_grid, ecl_rft, zonemap=None
):
    """
    Populate a list of trajectory points, that only contain UTM coordinates
    for a well-path, with (i,j,k) indices corresponding to a given Eclipse grid,
    and simulated pressure at the given date in the cells.

    Args:
        well (str): Eclipse-name of well
        date (datetime.date): Date for which the RFT observation should be taken,
            must correspond to a date in the Eclipse RFT binary output
        trajectory_points (list of points): Representing the wellpath in UTM at which
            the real life RFT observations points are valid.
        ecl_grid (libecl Grid object):
        ecl_rft (libecl RFT object):
        zonemap (dict):

    Returns:
        The list of trajectory points, augmented with i,j,k, simulated pressure
        and the zone name if zonemap is provided.
    """
    try:
        rft = ecl_rft.get(well, date)
    except KeyError:
        logger.error(
            (
                "Forward model script gendata_rft.py: "
                "No RFT data written for well ({}) and date ({})"
            ).format(well, date)
        )
        return []

    for point in trajectory_points:
        point.set_ijk(
            ecl_grid.find_cell(point.utm_x, point.utm_y, point.true_vertical_depth)
        )
        point.update_pressure_from_rft(rft)
        point.validate_zone(zonemap)

    return trajectory_points


def run(
    well_times,
    trajectories,
    ecl_grid,
    ecl_rft,
    zonemap=None,
    csvfile=None,
    outputdirectory=".",
):
    dframes = []

    if not well_times:
        raise ValueError("No RFT data requested")

    for well, time, report_step in well_times:
        logger.debug(
            "Collecting RFT for well: {} at date: {}, report step: {}".format(
                well, time, report_step
            )
        )

        trajectory_points = _populate_trajectory_points(
            well, time, trajectories[well], ecl_grid, ecl_rft, zonemap
        )

        if trajectory_points:
            # Aggregate the same data to a dataframe,
            # each trajectory tagged by well and time:
            trajectory_df = trajectory_points.to_dataframe(zonemap=zonemap).assign(
                well=well, time=time
            )

            # Write trajectory and associated data to three  files,
            # individual pr. well and time, this is for GEN_DATA to pick up.
            _write_gen_data_files(
                trajectory_df,
                os.path.join(outputdirectory, "RFT_{}_{}".format(well, report_step)),
            )

            # Aggregate dataframe for all wells and report steps.
            dframes.append(trajectory_df)

    if csvfile is not None:
        pd.concat(dframes, ignore_index=True, sort=False).to_csv(csvfile, index=None)

    if len(dframes) < len(well_times):
        raise ValueError("Failed to extract requested RFT data")
