import logging

logger = logging.getLogger(__name__)


def _write_points(points, fname):
    with open(fname, "w+") as fh:
        fh.write("\n".join([str(p.get_pressure()) for p in points]))
    logger.info("Forward model script gendata_rft.py: Wrote file {}".format(fname))

    with open(fname + "_active", "w+") as fh:
        fh.write("\n".join([str(int(p.is_active())) for p in points]))


def _write_inactive_info(points, fname, zonemap=None):
    with open(fname + "_inactive_info", "w+") as fh:
        fh.write(
            "\n".join([p.inactive_info(zonemap) for p in points if not p.is_active()])
        )


def _populate_trajectory_points(
    well, date, trajectory_points, ecl_grid, ecl_rft, zonemap=None
):
    try:
        rft = ecl_rft.get(well, date)
    except KeyError:
        logger.error(
            "Forward model script gendata_rft.py: No RFT data written for well ({}) and date ({})".format(
                well, date
            )
        )
        return []

    for point in trajectory_points:
        point.set_ijk(
            ecl_grid.find_cell(point.utm_x, point.utm_y, point.true_vertical_depth)
        )
        point.update_pressure_from_rft(rft)
        point.validate_zone(zonemap)

    return trajectory_points


def run(well_times, trajectories, ecl_grid, ecl_rft, zonemap=None):

    for well, time, report_step in well_times:
        logger.debug("Collecting RFT for well: {} at date: {}".format(well, time))

        trajectory_points = _populate_trajectory_points(
            well, time, trajectories[well], ecl_grid, ecl_rft, zonemap
        )
        if trajectory_points:
            fname = "RFT_{}_{}".format(well, report_step)

            _write_points(trajectory_points, fname)
            _write_inactive_info(trajectory_points, fname, zonemap)

    with open("GENDATA_RFT.OK", "w") as fh:
        fh.write("GENDATA RFT completed OK")
