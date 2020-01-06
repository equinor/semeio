import itertools
import argparse
import pytest

from semeio.jobs.rft.trajectory import TrajectoryPoint, Trajectory
from semeio.jobs.rft.zonemap import ZoneMap


@pytest.mark.parametrize(
    "line, expected_zone", [("0 1 2.2 3", None), ("0 1 2.2 3 zone", "zone"),]
)
def test_load_from_line(line, expected_zone):
    point = TrajectoryPoint(*Trajectory.parse_trajectory_line(line))
    assert point.utm_x == 0
    assert point.utm_y == 1
    assert point.measured_depth == 2.2
    assert point.true_vertical_depth == 3
    assert point.zone == expected_zone


@pytest.mark.parametrize(
    "line, expected_error",
    [
        ("1", "Trajectory data file not on correct format"),
        ("a b c", "Trajectory data file not on correct format"),
        ("a b c d e", "Error: Failed to extract data from line"),
    ],
)
def test_invalid_load_from_line(line, expected_error):
    with pytest.raises(argparse.ArgumentTypeError) as err_context:
        TrajectoryPoint(*Trajectory.parse_trajectory_line(line))
    assert expected_error in err_context.value.args[0]


@pytest.mark.parametrize(
    "k_value, expected_validation", [(1, True), (3, False), (4, False)]
)
def test_validate_zone(k_value, expected_validation):
    zones_at_k_value = {1: ["Zone1", "Zone2"], 3: ["Zone3"]}
    zone_map = ZoneMap(zones_at_k_value)

    point = TrajectoryPoint(0, 1, 2, 3, "Zone1")
    point.set_ijk([0, 0, k_value])

    assert not point.valid_zone
    point.validate_zone(zone_map)

    assert point.valid_zone == expected_validation


def test_validate_non_existing_zone():
    zones_at_k_value = {1: ["Zone1", "Zone2"], 3: ["Zone3"]}
    zone_map = ZoneMap(zones_at_k_value)

    point = TrajectoryPoint(0, 1, 2, 3, "non existing")
    point.set_ijk((0, 0, 1))
    point.validate_zone(zone_map)
    assert not point.valid_zone


ijks = [None, (0, 0, 0)]
pressures = [None, 1]
validations = [False, True]


@pytest.mark.parametrize(
    "ijk, pressure, validation", itertools.product(ijks, pressures, validations)
)
def test_is_active(ijk, pressure, validation):

    point = TrajectoryPoint(0, 1, 2, 3, "Zone1")
    point.set_ijk(ijk)
    point.pressure = pressure
    point.valid_zone = validation
    assert all([ijk, pressure, validation]) == point.is_active()

    assert 1 if all([ijk, pressure, validation]) else -1 == point.get_pressure()


@pytest.fixture()
def initdir(tmpdir):
    tmpdir.chdir()
    valid_data = """
0 1 2.2 3
4 5 6 7 zone
"""
    tmpdir.join("valid_trajectories.txt").write(valid_data)
    comment = "-- this is a comment"
    valid_data = comment + "\n" + valid_data + comment
    tmpdir.join("valid_trajectories_with_comments.txt").write(valid_data)


@pytest.mark.parametrize(
    "fname", ["valid_trajectories.txt", "valid_trajectories_with_comments.txt"]
)
def test_load(initdir, fname):
    expected_utmxs = [0, 4]
    trajectory = Trajectory.load_from_file(fname)

    for expected_utmx, trajectorypoint in zip(
        expected_utmxs, trajectory.trajectory_points
    ):
        assert trajectorypoint.utm_x == expected_utmx


def test_non_existing_file():

    with pytest.raises(argparse.ArgumentTypeError) as err_context:
        Trajectory.load_from_file("non_existing")

    assert (
        "Warning: Trajectory file non_existing not found!" == err_context.value.args[0]
    )
