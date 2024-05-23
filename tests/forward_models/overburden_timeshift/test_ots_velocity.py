import pytest
import segyio
from resdata.grid import GridGenerator

from semeio.forward_models.overburden_timeshift.ots_res_surface import OTSResSurface
from semeio.forward_models.overburden_timeshift.ots_vel_surface import OTSVelSurface

from .ots_util import create_segy_file


@pytest.fixture(name="setup_spec")
@pytest.mark.usefixtures("tmpdir")
def fixture_setup_spec():
    spec = segyio.spec()
    spec.format = 5
    spec.sorting = 2
    spec.samples = range(0, 40, 4)
    spec.ilines = range(2)
    spec.xlines = range(2)
    yield spec


def test_create(setup_spec):
    spec = setup_spec
    vel = "TEST.segy"
    create_segy_file(vel, spec)
    grid = GridGenerator.create_rectangular(dims=(2, 2, 2), dV=(100, 100, 100))
    res_surface = OTSResSurface(grid=grid, above=0)

    OTSVelSurface(res_surface, vel)


def test_surface(setup_spec):
    spec = setup_spec
    vel = "TEST.segy"
    int_val = [50, 150]
    create_segy_file(vel, spec, xl=int_val, il=int_val, cdp_x=int_val, cdp_y=int_val)
    grid = GridGenerator.create_rectangular(dims=(2, 2, 2), dV=(100, 100, 100))
    res_surface = OTSResSurface(grid=grid, above=0)
    ots_s = OTSVelSurface(res_surface, vel)

    assert ots_s.x[0] == 50
    assert ots_s.y[0] == 50
    assert ots_s.z[0] == 0
    assert ots_s.nx == 2
    assert ots_s.ny == 2


def test_z3d(setup_spec):
    spec = setup_spec
    vel = "TEST.segy"
    int_val = [50, 150]
    create_segy_file(vel, spec, xl=int_val, il=int_val, cdp_x=int_val, cdp_y=int_val)
    grid = GridGenerator.create_rectangular(dims=(2, 2, 2), dV=(100, 100, 100))
    res_surface = OTSResSurface(grid=grid, above=0)
    ots_s = OTSVelSurface(res_surface, vel)

    assert ots_s.z3d.shape == (4, 2)
