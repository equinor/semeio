# pylint: disable=invalid-name
import datetime
import os
from collections import namedtuple

import pytest
import segyio
from resdata.grid import GridGenerator
from xtgeo import surface_from_file

from semeio.forward_models.overburden_timeshift.ots import OverburdenTimeshift

from .ots_util import create_init, create_restart, create_segy_file

PARMS = namedtuple(
    "Parms",
    [
        "seabed",
        "above",
        "youngs",
        "poisson",
        "rfactor",
        "mapaxes",
        "convention",
        "output_dir",
        "horizon",
        "vintages_export_file",
        "velocity_model",
        "eclbase",
    ],
)


@pytest.fixture(name="set_up")
def fixture_set_up():
    spec = segyio.spec()
    spec.format = 5
    spec.sorting = 2
    spec.samples = range(0, 40, 4)
    spec.ilines = range(2)
    spec.xlines = range(2)

    actnum = [0, 0, 0, 0, 1, 1, 1, 1]

    PARMS.output_dir = None
    PARMS.horizon = None
    PARMS.vintages_export_file = None
    PARMS.velocity_model = None
    PARMS.seabed = 10
    PARMS.above = 10
    PARMS.youngs = 0.5
    PARMS.poisson = 0.3
    PARMS.rfactor = 20
    PARMS.convention = 1
    PARMS.eclbase = "TEST"

    yield spec, actnum, PARMS


@pytest.mark.parametrize(
    "missing_file, expected_error",
    [
        ("TEST.INIT", 'Failed to open file "TEST.INIT"'),
        ("TEST.EGRID", "Loading grid from:TEST.EGRID failed"),
        ("TEST.UNRST", 'Failed to open file "TEST.UNRST"'),
    ],
)
@pytest.mark.usefixtures("setup_tmpdir")
def test_create_missing_ecl_file(set_up, missing_file, expected_error):
    _, _, params = set_up
    grid = GridGenerator.create_rectangular(dims=(10, 10, 10), dV=(1, 1, 1))

    grid.save_EGRID("TEST.EGRID")
    create_init(grid, "TEST")
    create_restart(grid, "TEST")

    os.remove(missing_file)
    with pytest.raises(IOError, match=expected_error):
        OverburdenTimeshift(
            params.eclbase,
            params.mapaxes,
            params.seabed,
            params.youngs,
            params.poisson,
            params.rfactor,
            params.convention,
            params.above,
            params.velocity_model,
        )


def test_create_invalid_input_missing_segy(set_up):
    _, _, parms = set_up

    grid = GridGenerator.create_rectangular(dims=(10, 10, 10), dV=(1, 1, 1))
    grid.save_EGRID("TEST.EGRID")

    create_init(grid, "TEST")
    create_restart(grid, "TEST")

    parms.velocity_model = "missing.segy"

    with pytest.raises(IOError, match="No such file or directory"):
        OverburdenTimeshift(
            parms.eclbase,
            parms.mapaxes,
            parms.seabed,
            parms.youngs,
            parms.poisson,
            parms.rfactor,
            parms.convention,
            parms.above,
            parms.velocity_model,
        )


@pytest.mark.parametrize(
    "config_item, value", [("velocity_model", "TEST.segy"), ("velocity_model", None)]
)
@pytest.mark.usefixtures("setup_tmpdir")
def test_create_valid(set_up, config_item, value):
    spec, _, params = set_up
    grid = GridGenerator.create_rectangular(dims=(10, 10, 10), dV=(1, 1, 1))

    grid.save_EGRID("TEST.EGRID")
    create_init(grid, "TEST")
    create_restart(grid, "TEST")

    # Testing individual items in config, so setting that value here:
    setattr(params, config_item, value)
    if params.velocity_model:
        create_segy_file(params.velocity_model, spec)

    OverburdenTimeshift(
        params.eclbase,
        params.mapaxes,
        params.seabed,
        params.youngs,
        params.poisson,
        params.rfactor,
        params.convention,
        params.above,
        params.velocity_model,
    )


@pytest.mark.usefixtures("setup_tmpdir")
def test_eval(set_up):
    spec, actnum, parms = set_up
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=actnum
    )

    grid.save_EGRID("TEST.EGRID")
    create_init(grid, "TEST")
    create_restart(grid, "TEST")

    parms.velocity_model = "TEST.segy"
    create_segy_file(parms.velocity_model, spec)

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )
    with pytest.raises(ValueError):
        ots.add_survey("S1", datetime.date(2000, 1, 15))

    vintage_pairs = [(datetime.date(1900, 1, 1), datetime.date(2010, 1, 1))]

    with pytest.raises(ValueError):
        ots.geertsma_ts_simple(vintage_pairs)

    vintage_pairs = [(datetime.date(2010, 1, 1), datetime.date(1900, 1, 1))]

    with pytest.raises(ValueError):
        ots.geertsma_ts_simple(vintage_pairs)

    vintage_pairs = [(datetime.date(2000, 1, 1), datetime.date(2010, 1, 1))]

    ots.geertsma_ts_simple(vintage_pairs)


@pytest.mark.usefixtures("setup_tmpdir")
def test_geertsma_TS_simple(set_up):
    spec, actnum, parms = set_up
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=actnum
    )

    create_restart(grid, "TEST")
    create_init(grid, "TEST")
    grid.save_EGRID("TEST.EGRID")

    parms.velocity_model = "TEST.segy"

    int_val = [50, 150]
    create_segy_file(
        parms.velocity_model, spec, xl=int_val, il=int_val, cdp_x=int_val, cdp_y=int_val
    )

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    vintage_pairs = [
        (datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)),
        (datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)),
    ]

    tshift = ots.geertsma_ts_simple(vintage_pairs)
    assert tshift[0][(0, 0)] == pytest.approx(-0.01006, abs=0.0001)

    parms.convention = -1
    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    vintage_pairs = [
        (datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)),
        (datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)),
    ]

    tshift = ots.geertsma_ts_simple(vintage_pairs)
    assert tshift[0][(0, 0)] == pytest.approx(0.01006, abs=0.0001)


@pytest.mark.usefixtures("setup_tmpdir")
def test_geertsma_TS_rporv(set_up):
    spec, actnum, parms = set_up
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=actnum
    )

    create_restart(grid, "TEST", rporv=[10 for i in range(grid.getNumActive())])
    create_init(grid, "TEST")
    grid.save_EGRID("TEST.EGRID")

    parms.velocity_model = "TEST.segy"

    int_val = [50, 150]
    create_segy_file(
        parms.velocity_model, spec, xl=int_val, il=int_val, cdp_x=int_val, cdp_y=int_val
    )

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    vintage_pairs = [
        (datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)),
        (datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)),
    ]

    tshift = ots.geertsma_ts_rporv(vintage_pairs)
    assert tshift[0][(0, 0)] == pytest.approx(0.0, abs=0.0001)


@pytest.mark.usefixtures("setup_tmpdir")
def test_geertsma_TS(set_up):
    spec, actnum, parms = set_up
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=actnum
    )

    create_restart(grid, "TEST")
    create_init(grid, "TEST")
    grid.save_EGRID("TEST.EGRID")

    parms.velocity_model = "TEST.segy"

    int_val = [50, 150]
    create_segy_file(
        parms.velocity_model, spec, xl=int_val, il=int_val, cdp_x=int_val, cdp_y=int_val
    )

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    vintage_pairs = [
        (datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)),
        (datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)),
    ]

    tshift = ots.geertsma_ts(vintage_pairs)

    assert tshift[0][(0, 0)] == pytest.approx(-0.00104, abs=0.0001)

    parms.convention = -1

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    vintage_pairs = [
        (datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)),
        (datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)),
    ]

    tshift = ots.geertsma_ts(vintage_pairs)
    assert tshift[0][(0, 0)] == pytest.approx(0.00104, abs=0.0001)


@pytest.mark.usefixtures("setup_tmpdir")
def test_dPV(set_up):
    _, actnum, parms = set_up
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=actnum
    )

    grid.save_EGRID("TEST.EGRID")
    create_restart(grid, "TEST")
    create_init(grid, "TEST")

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    vintage_pairs = [
        (datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)),
        (datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)),
    ]

    tshift = ots.dpv(vintage_pairs)
    assert tshift[0][(0, 0)] == pytest.approx(((20 - 10) * 1e6 + (0 - 0) * 1e6) / 1e9)
    assert tshift[0][(0, 1)] == pytest.approx(((20 - 10) * 1e6 + (0 - 0) * 1e6) / 1e9)

    assert tshift[1][(0, 0)] == pytest.approx(((25 - 20) * 1e6 + (0 - 0) * 1e6) / 1e9)
    assert tshift[1][(0, 1)] == pytest.approx(((25 - 20) * 1e6 + (0 - 0) * 1e6) / 1e9)

    parms.convention = -1

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )
    tshift_b_m = ots.dpv(vintage_pairs)
    assert tshift[0][(0, 0)] == pytest.approx(-tshift_b_m[0][(0, 0)])


@pytest.mark.usefixtures("setup_tmpdir")
def test_irap_surface(set_up):
    spec, actnum, parms = set_up
    grid = GridGenerator.create_rectangular(
        dims=(2, 2, 2), dV=(100, 100, 100), actnum=actnum
    )

    # with TestAreaContext("test_irap_surface"):
    create_restart(grid, "TEST")
    create_init(grid, "TEST")
    grid.save_EGRID("TEST.EGRID")

    parms.velocity_model = "TEST.segy"
    create_segy_file(parms.velocity_model, spec)

    ots = OverburdenTimeshift(
        parms.eclbase,
        parms.mapaxes,
        parms.seabed,
        parms.youngs,
        parms.poisson,
        parms.rfactor,
        parms.convention,
        parms.above,
        parms.velocity_model,
    )

    f_name = "irap.txt"
    # pylint: disable=protected-access
    s = ots._create_surface()
    s.to_file(f_name)
    s = surface_from_file(f_name, fformat="irap_binary")

    assert s.get_nx() == 2
    assert s.get_ny() == 2

    assert s.values.ravel().tolist() == [90.0, 90.0, 90.0, 90.0]
