import argparse
import os

import pandas as pd

from semeio.jobs.rft.utility import strip_comments


class TrajectoryPoint:
    """Represents a point along a wellpath. The point is
    determined by UTM x and y, measured depth along the wellpath, and
    true vertical depth.

    Points can be *active*, which means that (i,j,k) in a given grid
    is determined, there is simulated pressure available in that cell,
    and the cell in in the correct zone (mostly important for long
    horizontals).

    RKB for MD must match MD in Eclipse RFT data.
    """

    def __init__(self, utm_x, utm_y, measured_depth, true_vertical_depth, zone=None):
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.measured_depth = measured_depth
        self.true_vertical_depth = true_vertical_depth
        self.zone = zone
        self.grid_ijk = None
        self.pressure = None
        self.valid_zone = False

    def set_ijk(self, point):
        self.grid_ijk = point

    def validate_zone(self, zonemap=None):
        if self.zone is None:
            self.valid_zone = True
        elif self.grid_ijk:
            self.valid_zone = zonemap is None or zonemap.has_relationship(
                self.zone, self.grid_ijk[2]
            )

    def is_active(self):
        return (
            self.grid_ijk is not None and self.pressure is not None and self.valid_zone
        )

    def inactive_info(self, zonemap=None):
        if self.grid_ijk is None:
            return "TRAJECTORY_POINT_NOT_IN_GRID {}".format(str(self))
        if self.pressure is None:
            return "TRAJECTORY_POINT_NOT_IN_RFT {}".format(str(self))
        if not self.valid_zone and zonemap is not None:
            if self.grid_ijk[2] not in zonemap:
                return "ZONEMAP_MISSING_VALUE {} {} {}".format(
                    str(self), self.grid_ijk[2], None
                )

            return "ZONE_MISMATCH {} {} {}".format(
                str(self), self.grid_ijk[2], zonemap[self.grid_ijk[2]]
            )

    def get_pressure(self):
        if self.is_active():
            return self.pressure
        return -1

    def update_pressure_from_rft(self, rft):
        if self.grid_ijk:
            rftcell = rft.ijkget(self.grid_ijk)
            if rftcell:
                self.pressure = rftcell.pressure

    def __str__(self):
        return "(utm_x={utm_x}, utm_y={utm_y}, measured_depth={measured_depth})".format(
            utm_x=self.utm_x, utm_y=self.utm_y, measured_depth=self.measured_depth
        )


class Trajectory:
    """Represents a well trajectory as a list of TrajectoryPoints.

    Args:
        points (list): List of lists with data for the trajectory.
    """

    def __init__(self, points):
        self.trajectory_points = [TrajectoryPoint(*point) for point in points]

    def __getitem__(self, i):
        return self.trajectory_points[i]

    def __len__(self):
        return len(self.trajectory_points)

    def to_dataframe(self, zonemap=None):
        """Expose the trajectory data as a Pandas DataFrame.

        The data grid_ijk in self is a tuple, this is split into
        three distinct columns, i, j and k.

        The dataframe is sorted first by measured_depth, if not available
        it is sorted by true_vertical_depth. The original order from the
        input files is conserved in the column "order".

        Returns:
            pd.DataFrame
        """
        dframe = pd.DataFrame(data=[vars(point) for point in self.trajectory_points])
        dframe["is_active"] = [point.is_active() for point in self.trajectory_points]
        dframe["inactive_info"] = [
            point.inactive_info(zonemap) for point in self.trajectory_points
        ]

        dframe = Trajectory.split_tuple_column(
            dframe, tuplecolumn="grid_ijk", components=["i", "j", "k"]
        )

        # Conserve original order before sorting - original order from
        # well input files is needed by some export routines.
        dframe.index.name = "order"
        dframe = dframe.reset_index()

        if "measured_depth" in dframe:
            dframe = dframe.sort_values("measured_depth")
        elif "true_vertical_depth" in dframe:
            dframe = dframe.sort_values("true_vertical_depth")

        return dframe.dropna(how="all", axis="columns")

    @staticmethod
    def split_tuple_column(dframe, tuplecolumn="grid_ijk", components=None):
        """For a dataframe with a column containing a tuple
        datatype, split the tuple into the components [I, J, K] as new columns.

        Args:
            dframe (pd.DataFrame): A dataframe with a tuple column
            tuplecolumn (str): Column name with tuples. Defaults "grid_ijk"
            components (list of str): Name of components in tuple, default
                ["i", "j", "k"]
        Returns:
            pd.DataFrame, one column removed, and 3 (if defaults) new columns added.
                Column order is undefined.
        """
        if components is None:
            components = ["i", "j", "k"]

        non_nulls = ~dframe[tuplecolumn].isnull()
        if not non_nulls.any():
            return dframe
        tuplelengths = {len(value) for value in dframe.loc[non_nulls, tuplecolumn]}
        if len(tuplelengths) != 1:
            raise ValueError(
                "Uneven tuple lengths {} in non-null dataframe data".format(
                    tuplelengths
                )
            )
        if list(tuplelengths)[0] != len(components):
            raise ValueError("Mismatch between tuple length and given column names")

        ijk_df = pd.DataFrame(
            data=dframe.loc[non_nulls, tuplecolumn].tolist(),
            index=dframe.index[non_nulls],
            columns=components,
        )
        return pd.concat(
            [dframe.drop(tuplecolumn, axis="columns"), ijk_df],
            axis="columns",
            ignore_index=False,
            sort=False,
        )

    @staticmethod
    def parse_trajectory_line(line):
        """Reads a text line with four floating point values (utm_x, utm_y, md, and tvd)
        and an optional string being the zone name.

        Returns:
            List of four floats or a list of four floats and a string.
        """
        point = line.split()
        if len(point) < 4 or len(point) > 5:
            raise argparse.ArgumentTypeError(
                (
                    "Trajectory data file not on correct format: "
                    "'utm_x utm_y md tvd <zone>' - zone is optional"
                )
            )

        try:
            floats = [float(v) for v in point[:4]]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Error: Failed to extract data from line {line}".format(line=line)
            )

        zone = point[4].strip() if len(point) == 5 else None
        return floats + [zone]

    @classmethod
    def load_from_file(cls, filepath):
        """Initialize a Trajectory from a text file describing a well path."""
        trajectory_points = []

        filename = os.path.join(filepath)
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(
                "Warning: Trajectory file {filename} not found!".format(
                    filename=filename
                )
            )

        with open(filename, "r") as f:
            trajectory_lines = f.readlines()

        trajectory_lines = [strip_comments(line) for line in trajectory_lines]

        for line in trajectory_lines:
            if not line:
                continue
            point = Trajectory.parse_trajectory_line(line)
            trajectory_points.append(point)

        return cls(trajectory_points)
