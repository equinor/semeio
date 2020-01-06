import argparse
import os

from semeio.jobs.rft.utility import strip_comments


class TrajectoryPoint:
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


class Trajectory:
    def __init__(self, points):
        self.trajectory_points = [TrajectoryPoint(*point) for point in points]

    def __getitem__(self, i):
        return self.trajectory_points[i]

    def __len__(self):
        return len(self.trajectory_points)

    @staticmethod
    def parse_trajectory_line(line):
        point = line.split()
        if len(point) < 4 or len(point) > 5:
            raise argparse.ArgumentTypeError(
                "Trajectory data file not on correct format: 'utm_x utm_y md tvd <zone>' - zone is optional"
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

        trajectory_lines = [strip_comments(l) for l in trajectory_lines]

        for line in trajectory_lines:
            if not line:
                continue
            point = Trajectory.parse_trajectory_line(line)
            trajectory_points.append(point)

        return cls(trajectory_points)
