import os

import pandas as pd
from resdata.rft import ResdataRFTCell

from semeio.forward_models.rft.utility import strip_comments


class TrajectoryPoint:
    """Represents a point along a wellpath. The point is
    determined by UTM x and y, measured depth along the wellpath, and
    true vertical depth.

    Points can be *active*, which means that (i,j,k) (zero-indexed) in a given
    grid is determined, there is simulated pressure available in that cell, and
    the cell in in the correct zone (mostly important for long horizontals).

    RKB for MD must match MD in Eclipse RFT data.

    Args:
        utm_x (float)
        utm_y (float)
        measured_depth (float): Depth along wellpath. RKB (rotary kelly bushing)
            must be compatible with Eclipse setup.
        true_vertical_depth (float)
        zone (str)
    """

    def __init__(self, utm_x, utm_y, measured_depth, true_vertical_depth, zone=None):
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.measured_depth = measured_depth
        self.true_vertical_depth = true_vertical_depth
        self.zone = zone
        self.grid_ijk = None  # tuple
        self.pressure = None
        self.swat = None
        self.sgas = None
        self.soil = None
        self.valid_zone = False

    def set_ijk(self, point):
        """Set the ijk-tuple for the point, relating the UTM-coordinates to
        ijk-coordinates in a specific Eclipse grid.

        Args:
            point (tuple): 3-tuple with ijk-integers, zero-indexed.
        """
        self.grid_ijk = point

    def validate_zone(self, zonemap=None):
        """Update the internal valid_zone property determining if
        the point can be validated. If the point is not initialized
        to be in a specific zone and has a well-defined k-index, the
        validation always succeeds.

        Args:
            zonemap (Zonemap)
        """
        if self.zone is None:
            self.valid_zone = True
        elif self.grid_ijk:
            self.valid_zone = zonemap is None or zonemap.has_relationship(
                self.zone, self.grid_ijk[2]
            )

    def is_active(self):
        """Determines if the point is regarded as active, and thus usable
        in a history match setting - meaning there is a simulated pressure
        value available and the zone is valid"""
        return (
            self.grid_ijk is not None and self.pressure is not None and self.valid_zone
        )

    def inactive_info(self, zonemap=None):
        """Provides a string explaining why a point is not active.

        Returns None for active points.

        Args:
            zonemap (Zonemap)

        Returns:
            str
        """
        if self.grid_ijk is None:
            return f"TRAJECTORY_POINT_NOT_IN_GRID {str(self)}"
        if self.pressure is None:
            return f"TRAJECTORY_POINT_NOT_IN_RFT {str(self)}"
        if not self.valid_zone and zonemap is not None:
            if self.grid_ijk[2] not in zonemap:
                return f"ZONEMAP_MISSING_VALUE {str(self)} {self.grid_ijk[2]} {None}"

            return (
                f"ZONE_MISMATCH {str(self)} "
                f"{self.grid_ijk[2]} "
                f"{zonemap[self.grid_ijk[2]]}"
            )
        return None

    def get_pressure(self):
        """Returns the simulated pressure for the point, or -1 if
        no simulated pressure is available

        Returns:
            float
        """
        if self.is_active():
            return self.pressure
        return -1

    def update_simdata_from_rft(self, rftfile):
        """Fetch simulated data from an Eclipse simulation by looking up
        binary RFT files. This requires the point to have the ijk-tuple
        set upfront.

        Args:
            rftfile (EclRFTFile)
        """
        if self.grid_ijk:
            rftcell = rftfile.ijkget(self.grid_ijk)
            if rftcell:
                self.pressure = rftcell.pressure
                # The rftcell object is either a EclPLTCell instance, then it
                # contains only pressure information, or it is an ResdataRFTCell
                # instance, then it also contains saturation information. The
                # type of cell written to the binary rftfile is determined by
                # configuration settings in the Eclipse deck.
                if isinstance(rftcell, ResdataRFTCell):
                    self.swat = rftcell.swat
                    self.sgas = rftcell.sgas
                    if self.sgas is not None and self.sgas > -1:
                        # ResdataRFTCell will return -1 as an invalid value.
                        self.soil = 1 - self.swat - self.sgas
                    else:
                        # Two-phase Eclipse runs
                        self.soil = 1 - self.swat

    def __str__(self):
        return (
            f"(utm_x={self.utm_x}, "
            f"utm_y={self.utm_y}, "
            f"measured_depth={self.measured_depth})"
        )


class Trajectory:
    """Represents a well trajectory as a list of TrajectoryPoints.

    Args:
        points (list): List of lists with (textual) data for the trajectory.
    """

    def __init__(self, points):
        self.trajectory_points = [TrajectoryPoint(*point) for point in points]

    def __getitem__(self, i):
        return self.trajectory_points[i]

    def __len__(self):
        return len(self.trajectory_points)

    def to_dataframe(self, zonemap=None):
        """Expose the trajectory data as a Pandas DataFrame.

        The data grid_ijk in self is a tuple, this is split into three distinct
        columns, i, j and k. The i-, j- and k-columns from this function are
        1-indexed.

        The dataframe is sorted first by measured_depth, if not available
        it is sorted by true_vertical_depth. The original order from the
        input files is conserved in the column "order".

        Args:
            zonemap (Zonemap)

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

        # Convert from libecl 0-indexed coordinates to Eclipse-style 1-indexed:
        if {"i", "j", "k"}.issubset(dframe.columns):
            dframe[["i", "j", "k"]] += 1

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

        non_nulls = ~dframe[tuplecolumn].isna()
        if not non_nulls.any():
            return dframe
        tuplelengths = {len(value) for value in dframe.loc[non_nulls, tuplecolumn]}
        if len(tuplelengths) != 1:
            raise ValueError(
                f"Uneven tuple lengths {tuplelengths} in non-null dataframe data"
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
            raise ValueError(
                "Trajectory data file not on correct format: "
                "'utm_x utm_y md tvd <zone>' - zone is optional"
            )

        try:
            floats = [float(v) for v in point[:4]]
        except ValueError as err:
            raise ValueError(
                f"Error: Failed to extract data from line {line}. Expected the format "
                "'utm_x utm_y md tvd zone' - zone is optional, where utm coordinates, "
                "md and tvd are numbers"
            ) from err

        zone = point[4].strip() if len(point) == 5 else None
        return floats + [zone]

    @classmethod
    def load_from_file(cls, filepath):
        """Initialize a Trajectory from a text file describing a well path."""
        trajectory_points = []

        filename = os.path.join(filepath)
        if not os.path.isfile(filename):
            raise OSError(f"Trajectory file {filename} not found!")

        with open(filename, encoding="utf8") as file_handle:
            trajectory_lines = file_handle.readlines()

        trajectory_lines = [strip_comments(line) for line in trajectory_lines]

        for line in trajectory_lines:
            if not line:
                continue
            point = Trajectory.parse_trajectory_line(line)
            trajectory_points.append(point)

        return cls(trajectory_points)
