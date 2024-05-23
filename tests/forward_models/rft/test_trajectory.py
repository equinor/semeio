import itertools
from pathlib import Path

import pandas as pd
import pytest

from semeio.forward_models.rft.trajectory import Trajectory, TrajectoryPoint
from semeio.forward_models.rft.zonemap import ZoneMap


@pytest.mark.parametrize(
    "line, expected_zone", [("0 1 2.2 3", None), ("0 1 2.2 3 zone", "zone")]
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
    with pytest.raises(ValueError, match=expected_error):
        TrajectoryPoint(*Trajectory.parse_trajectory_line(line))


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

    assert all([ijk, pressure, validation]) or point.get_pressure() == -1


@pytest.fixture()
def initdir(tmpdir):
    tmpdir.chdir()
    valid_data = """
0 1 2.2 3
4 5 6 7 zone
"""
    # (utm_x utm_y md tvd zone_name)
    tmpdir.join("valid_trajectories.txt").write(valid_data)
    comment = "-- this is a comment"
    valid_data = comment + "\n" + valid_data + comment
    tmpdir.join("valid_trajectories_with_comments.txt").write(valid_data)


@pytest.mark.parametrize(
    "fname", ["valid_trajectories.txt", "valid_trajectories_with_comments.txt"]
)
@pytest.mark.usefixtures("initdir")
def test_load(fname):
    expected_utmxs = [0, 4]
    trajectory = Trajectory.load_from_file(fname)

    for expected_utmx, trajectorypoint in zip(
        expected_utmxs, trajectory.trajectory_points
    ):
        assert trajectorypoint.utm_x == expected_utmx


@pytest.mark.parametrize(
    "fname", ["valid_trajectories.txt", "valid_trajectories_with_comments.txt"]
)
@pytest.mark.usefixtures("initdir")
def test_dframe_trajectory(fname):
    """Test dataframe representation of a trajectory not having
    any attached Eclipse simulation results"""
    dframe = Trajectory.load_from_file(fname).to_dataframe()

    assert isinstance(dframe, pd.DataFrame)

    # Dataframe lengths should be the same as number of non-empty lines
    # in txt input:

    assert len(dframe) == len(
        [
            line
            for line in Path(fname).read_text(encoding="utf-8").split("\n")
            if line.strip() and not line.strip().startswith("--")
        ],
    )

    # Check that we have the input columns which defines the trajectory:
    assert {"utm_x", "utm_y", "measured_depth", "true_vertical_depth", "zone"}.issubset(
        set(dframe.columns)
    )

    # grid_ijk is a temp column, never to be present in output
    assert "grid_ijk" not in dframe
    # and since there is no Eclipse results attached, we can't have these either:
    assert "i" not in dframe
    assert "j" not in dframe
    assert "k" not in dframe

    # Check casing for column names, ensuring consistency in this particular
    # dataframe:
    assert list(dframe.columns) == [colname.lower() for colname in dframe.columns]

    # pressure should not be there when there is no data for it
    # (no Eclipse simulation results attached in this particular test function)
    assert "pressure" not in dframe

    # These columns should be in place, to signify there is not data for them.
    # (Eclipse simulation results would be needed for any of these to be True)
    assert set(dframe["valid_zone"]) == {False}
    assert set(dframe["is_active"]) == {False}

    # Check dataframe sorting:
    assert (
        dframe.sort_values("measured_depth")["measured_depth"]
        == dframe["measured_depth"]
    ).all()


@pytest.mark.parametrize(
    "dframe, tuplecolumn, components",
    [
        (pd.DataFrame(columns=["grid_ik"], data=[[(1, 2)]]), "grid_ik", ["I", "J"]),
        (pd.DataFrame(columns=["grid_i"], data=[[(1,)]]), "grid_i", ["I"]),
        (
            pd.DataFrame(columns=["grid_i"], data=[[[1, 2, 3]]]),
            "grid_i",
            ["I", "J", "K"],
        ),
        (
            pd.DataFrame(columns=["grid_i"], data=[[(1, 2)], [(3, 4)]]),
            "grid_i",
            ["I", "J"],
        ),
        (
            pd.DataFrame(columns=["grid_i"], data=[[(1, 2)], [(3, 4)]]),
            "grid_i",
            ["I", "J"],
        ),
        (
            pd.DataFrame(
                columns=["grid_i", "extra"], data=[[(1, 2), "foo"], [(3, 4), "bar"]]
            ),
            "grid_i",
            ["I", "J"],
        ),
    ],
)
def test_tuple_column_splitter(dframe, tuplecolumn, components):
    splitdf = Trajectory.split_tuple_column(
        dframe, tuplecolumn=tuplecolumn, components=components
    )

    assert tuplecolumn in dframe  # Ensure we have not touched the input
    assert tuplecolumn not in splitdf
    assert len(dframe) == len(splitdf)
    assert {val for tup in dframe[tuplecolumn] for val in tup} == {
        val for tup in splitdf[components].to_numpy() for val in tup
    }
    for comp in components:
        assert comp in splitdf
    assert len(splitdf.columns) == len(dframe.columns) - 1 + len(components)


@pytest.mark.parametrize(
    "raises, dframe, tuplecolumn, components",
    [
        (
            ValueError,
            pd.DataFrame(columns=["grid_ik"], data=[[(1, 2)]]),
            "grid_ik",
            ["I", "J", "K"],
        ),
        (
            ValueError,
            pd.DataFrame(columns=["grid_ik"], data=[[(1, 2)], [(1, 2, 3)]]),
            "grid_ik",
            ["I", "J", "K"],
        ),
        (
            ValueError,
            pd.DataFrame(columns=["grid_ik"], data=[[(1, 2)], [(1, 2, 3)]]),
            "grid_ik",
            ["I", "J"],
        ),
    ],
)
def test_tuple_column_splitter_errors(raises, dframe, tuplecolumn, components):
    with pytest.raises(raises):
        Trajectory.split_tuple_column(
            dframe, tuplecolumn=tuplecolumn, components=components
        )


def test_tuple_column_splitter_explicit():
    # Checks that None-rows are handled, and default parameters for split_tuple_column()
    dframe = Trajectory.split_tuple_column(
        pd.DataFrame(columns=["grid_ijk"], data=[[None], [(1, 2, 3)]])
    )
    assert len(dframe) == 2
    assert {"i", "j", "k"}.issubset(set(dframe.columns))


def test_non_existing_file():
    with pytest.raises(IOError, match="Trajectory file non_existing not found!"):
        Trajectory.load_from_file("non_existing")
