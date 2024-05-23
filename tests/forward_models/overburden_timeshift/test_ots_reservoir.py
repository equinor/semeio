import os

import numpy as np
import pytest
from resdata.grid import Grid, GridGenerator

from semeio.forward_models.overburden_timeshift.ots_res_surface import OTSResSurface


def get_source_ert(grid):
    # pylint: disable=invalid-name
    x = np.zeros(grid.getNumActive(), np.float64)
    y = np.zeros(grid.getNumActive(), np.float64)
    z = np.zeros(grid.getNumActive(), np.float64)
    v = np.zeros(grid.getNumActive(), np.float64)
    for i in range(grid.getNumActive()):
        (x[i], y[i], z[i]) = grid.get_xyz(active_index=i)
        v[i] = grid.cell_volume(active_index=i)

    return x, y, z, v


def test_res_surface(ots_tmpdir_enter):
    eclcase_dir = ots_tmpdir_enter
    grid_path = os.path.join(eclcase_dir, "NORNE_ATW2013.EGRID")
    grid = Grid(grid_path, apply_mapaxes=False)

    rec = OTSResSurface(grid=grid, above=100)

    assert rec.nx == 46
    assert rec.ny == 112

    assert pytest.approx(np.min(rec.x)) == 453210.38
    assert pytest.approx(np.max(rec.x)) == 465445.16

    assert pytest.approx(np.min(rec.y)) == 7316018.5
    assert pytest.approx(np.max(rec.y)) == 7330943.5

    assert pytest.approx(np.min(rec.z)) == 2177.6484
    assert pytest.approx(np.max(rec.z)) == 3389.567


@pytest.mark.usefixtures("tmpdir")
def test_surface():
    # pylint: disable=invalid-name
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=[0, 0, 0, 0, 1, 1, 1, 1]
    )

    surface = OTSResSurface(grid=grid)

    z = np.ones(4) * 100
    assert np.all(z == surface.z)


@pytest.mark.usefixtures("tmpdir")
def test_surface_above():
    # pylint: disable=invalid-name
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=[0, 0, 0, 0, 1, 1, 1, 1]
    )

    surface = OTSResSurface(grid=grid, above=10)

    z = np.ones(4) * (100 - 10)
    assert np.all(z == surface.z)
