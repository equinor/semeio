"""Test the Trajectory and TrajectoryPoint classes with binary Eclipse data"""

import datetime

import numpy as np
import pytest
from resdata.grid import Grid
from resdata.rft import ResdataRFTFile

from semeio.forward_models.rft.trajectory import Trajectory, TrajectoryPoint
from tests.forward_models.rft import conftest

ECL_BASE_REEK = conftest.get_ecl_base_reek()
ECL_BASE_NORNE = conftest.get_ecl_base_norne()


@pytest.mark.usefixtures("reek_data")
def test_update_simdata_from_rft_reek():
    """Test data extraction from the binary Eclipse files
    for a single well point using TrajectoryPoint.update_simdata_from_rft()"""

    grid = Grid(ECL_BASE_REEK + ".EGRID")
    rft = ResdataRFTFile(ECL_BASE_REEK + ".RFT")
    rft_well_date = rft.get("OP_1", datetime.date(2000, 2, 1))

    # A trajectory point for an active cell in reek:
    point = TrajectoryPoint(462608.57, 5934210.96, 1624.38, 1624.38)
    assert point.grid_ijk is None
    assert point.pressure is None
    assert point.swat is None
    assert point.sgas is None
    assert point.soil is None
    point.set_ijk(grid.find_cell(point.utm_x, point.utm_y, point.true_vertical_depth))
    assert point.grid_ijk == (28, 27, 7)
    point.update_simdata_from_rft(rft_well_date)
    assert np.isclose(point.pressure, 304.37)
    assert np.isclose(point.swat, 0.151044)
    assert np.isclose(point.soil, 1 - 0.151044)
    assert np.isclose(point.sgas, 0.0)

    # Construct a Trajectory from the point
    traj = Trajectory([])
    traj.trajectory_points = [point]  # (can't initialize from list of points)
    dframe = traj.to_dataframe()
    assert {"i", "j", "k", "pressure", "soil", "sgas", "swat"}.issubset(set(dframe))


@pytest.mark.usefixtures("norne_data")
def test_update_simdata_from_rft_norne():
    """Similar test as the reek version, but the Norne RFT file
    does not contain saturations, and in libecl terms contains EclPLTCell
    as opposed to EclRFTCell as in Reek"""

    grid = Grid(ECL_BASE_NORNE + ".EGRID")
    rft = ResdataRFTFile(ECL_BASE_NORNE + ".RFT")
    rft_well_date = rft.get("C-3H", datetime.date(1999, 5, 4))

    # A trajectory point for an active cell in Norne, picked
    # from a line in gendata_rft_input_files/C-3H.txt
    point = TrajectoryPoint(
        455752.59771598293, 7321015.949386452, 2785.78173828125, 2785.78173828125
    )
    assert point.grid_ijk is None
    assert point.pressure is None
    assert point.swat is None
    assert point.sgas is None
    assert point.soil is None
    point.set_ijk(grid.find_cell(point.utm_x, point.utm_y, point.true_vertical_depth))
    assert point.grid_ijk == (8, 12, 20)  # Zero-indexed integers.
    point.update_simdata_from_rft(rft_well_date)
    # There is no saturation data in the Norne binary output, then these
    # should be None
    assert point.swat is None
    assert point.sgas is None
    assert point.soil is None

    # Construct a Trajectory from the point
    traj = Trajectory([])
    traj.trajectory_points = [point]  # (can't initialize from list of points)
    dframe = traj.to_dataframe()
    assert {"i", "j", "k", "pressure"}.issubset(set(dframe))
    assert "swat" not in dframe


def test_update_simdata_outside_grid():
    grid = Grid(ECL_BASE_REEK + ".EGRID")
    rft = ResdataRFTFile(ECL_BASE_REEK + ".RFT")
    rft_well_date = rft.get("OP_1", datetime.date(2000, 2, 1))

    # A point outside the grid:
    point = TrajectoryPoint(45000, 60000000, 1, 1)
    point.set_ijk(grid.find_cell(point.utm_x, point.utm_y, point.true_vertical_depth))
    assert point.grid_ijk is None  # There is no Exception raised by set_ijk()

    point.update_simdata_from_rft(rft_well_date)
    assert point.pressure is None  # Since we are outside the grid.

    # Construct a Trajectory from the point
    traj = Trajectory([])
    traj.trajectory_points = [point]  # (can't initialize from list of points)
    dframe = traj.to_dataframe()
    assert not set(dframe).intersection({"i", "j", "k", "pressure", "swat"})


def test_update_simdata_outside_well():
    grid = Grid(ECL_BASE_REEK + ".EGRID")
    rft = ResdataRFTFile(ECL_BASE_REEK + ".RFT")
    rft_well_date = rft.get("OP_1", datetime.date(2000, 2, 1))

    # A point in the grid, but not related to the well
    point = TrajectoryPoint(462825.55, 5934025.52, 1623.19, 1623.19)
    point.set_ijk(grid.find_cell(point.utm_x, point.utm_y, point.true_vertical_depth))
    # NB: grid_ijk ints start at zero, ResInsight and ecl2df report this as (29, 29, 7)
    assert point.grid_ijk == (28, 28, 6)
    point.update_simdata_from_rft(rft_well_date)
    assert point.pressure is None
    assert point.swat is None
    assert point.sgas is None
    assert point.soil is None

    # Construct a Trajectory from the point
    traj = Trajectory([])
    traj.trajectory_points = [point]  # (can't initialize from list of points)
    dframe = traj.to_dataframe()
    assert {"i", "j", "k"}.issubset(set(dframe))
    assert not set(dframe).intersection({"pressure", "swat", "soil", "sgas"})
