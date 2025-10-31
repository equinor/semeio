import os
from collections.abc import Iterator
from typing import Any, Self, TypeAlias

import pandas as pd
from resdata.rft import ResdataRFT, ResdataRFTCell

from semeio.forward_models.rft.utility import strip_comments
from semeio.forward_models.rft.zonemap import ZoneMap

IJKCoordinates: TypeAlias = tuple[int, int, int]


class TrajectoryPoint:
    """Represents a point along a wellpath.

    The point is determined by UTM x and y, measured depth along the wellpath,
    and true vertical depth.

    Points can be *active*, which means that (i,j,k) (zero-indexed) in a given
    grid is determined, there is simulated pressure available in that cell, and
    the cell in the correct zone (mostly important for long horizontals).

    RKB for MD must match MD in Eclipse RFT data.

    Args:
        utm_x: The utm coordinate in x direction
        utm_y: The utm coordinate in y direction
        measured_depth: Depth along wellpath. RKB (rotary kelly bushing)
            must be compatible with Eclipse setup.
        true_vertical_depth: Depth in distance from surface.
        zone: The name of the zone the point belongs to.
    """

    def __init__(
        self,
        utm_x: float,
        utm_y: float,
        measured_depth: float,
        true_vertical_depth: float,
        zone: str | None = None,
    ) -> None:
        self.utm_x: float = utm_x
        self.utm_y: float = utm_y
        self.measured_depth: float = measured_depth
        self.true_vertical_depth: float = true_vertical_depth
        self.zone: str | None = zone
        self.grid_ijk: IJKCoordinates | None = None  # tuple
        self.pressure: float | None = None
        self.swat: float | None = None
        self.sgas: float | None = None
        self.soil: float | None = None
        self.valid_zone: bool = False

    def set_ijk(self, point: IJKCoordinates | None) -> None:
        """Set the ijk-coordinates for the point, relating the UTM-coordinates to
        ijk-coordinates in a specific Eclipse grid.
        """
        self.grid_ijk = point

    def validate_zone(self, zonemap: ZoneMap | None = None) -> None:
        """Update the internal valid_zone property determining if
        the point can be validated. If the point is not initialized
        to be in a specific zone and has a well-defined k-index, the
        validation always succeeds.
        """
        if self.zone is None:
            self.valid_zone = True
        elif self.grid_ijk:
            self.valid_zone = zonemap is None or zonemap.has_relationship(
                self.zone, self.grid_ijk[2]
            )

    def is_active(self) -> bool:
        """Determines if the point is regarded as active, and thus usable
        in a history match setting - meaning there is a simulated pressure
        value available and the zone is valid"""
        return (
            self.grid_ijk is not None and self.pressure is not None and self.valid_zone
        )

    def inactive_info(self, zonemap: ZoneMap | None = None) -> str | None:
        """Provides a string explaining why a point is not active.

        Returns None for active points.
        """
        if self.grid_ijk is None:
            return f"TRAJECTORY_POINT_NOT_IN_GRID {self!s}"
        if self.pressure is None:
            return f"TRAJECTORY_POINT_NOT_IN_RFT {self!s}"
        if not self.valid_zone and zonemap is not None:
            if self.grid_ijk[2] not in zonemap:
                return f"ZONEMAP_MISSING_VALUE {self!s} {self.grid_ijk[2]} {None}"

            return (
                f"ZONE_MISMATCH {self!s} {self.grid_ijk[2]} {zonemap[self.grid_ijk[2]]}"
            )
        return None

    def get_pressure(self) -> float | None:
        """Returns the simulated pressure for the point, or -1 if
        no simulated pressure is available
        """
        if self.is_active():
            return self.pressure
        return -1

    def update_simdata_from_rft(self, rftfile: ResdataRFT) -> None:
        """Fetch simulated data from an Eclipse simulation by looking up
        binary RFT files. This requires the point to have the ijk-coordinates
        set upfront.
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

    def __str__(self) -> str:
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

    def __init__(self, points: Any) -> None:  # noqa: ANN401
        self.trajectory_points: list[TrajectoryPoint] = [
            TrajectoryPoint(*point) for point in points
        ]

    def __getitem__(self, i: int) -> TrajectoryPoint:
        return self.trajectory_points[i]

    def __len__(self) -> int:
        return len(self.trajectory_points)

    def __iter__(self) -> Iterator[TrajectoryPoint]:
        return iter(self.trajectory_points)

    def to_dataframe(self, zonemap: ZoneMap | None = None) -> pd.DataFrame:
        """Expose the trajectory data as a Pandas DataFrame.

        The data grid_ijk in self is a tuple, this is split into three distinct
        columns, i, j and k. The i-, j- and k-columns from this function are
        1-indexed.

        The dataframe is sorted first by measured_depth, if not available
        it is sorted by true_vertical_depth. The original order from the
        input files is conserved in the column "order".
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
    def split_tuple_column(
        dframe: pd.DataFrame,
        tuplecolumn: str = "grid_ijk",
        components: list[str] | None = None,
    ) -> pd.DataFrame:
        """For a dataframe with a column containing a tuple
        datatype, split the tuple into the components [I, J, K] as new columns.

        Args:
            dframe: A dataframe with a tuple column
            tuplecolumn : Column name with tuples. Defaults "grid_ijk"
            components: Name of components in tuple, default ["i", "j", "k"]
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
        if next(iter(tuplelengths)) != len(components):
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
    def parse_trajectory_line(line: str) -> list[float | str | None]:
        """Reads a text line with four floating point values (utm_x, utm_y, md, and tvd)
        and an optional string being the zone name.

        Returns:
            List of four floats or a list of four floats and a string.

        Raises:
            ValueError: For invalid lines (too few/many values or non-float values).
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
        return [*floats, zone]

    @classmethod
    def load_from_file(cls, filepath: str) -> Self:
        """Initialize a Trajectory from a text file describing a well path."""
        trajectory_points: list[Any] = []

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
