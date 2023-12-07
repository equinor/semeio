import os

import pytest
import xtgeo
import yaml
from resdata.grid import Grid

from semeio.forward_models.overburden_timeshift.ots import ots_run

from .ots_util import mock_segy

# pylint: disable=too-many-statements, duplicate-code


@pytest.mark.parametrize(
    "res_scale, size_scale, pos_shift, results, surf_res",
    [
        # all scales or shifts are relative to the input grid file
        # segy file resolution, volume size, position of segy volume
        ((1, 1, 1), (1, 1), (0, 0), (True, True, False), (112, 46)),
        ((3, 3, 3), (2, 2), (0.5, 0.5), (True, False, True), (325, 138)),
        ((1, 1, 1), (2, 2), (0.5, 0.5), (True, False, False), (112, 46)),
        ((1, 1, 1), (2, 2), (-0.5, -0.5), (False, True, False), (112, 46)),
    ],
)
def test_ots_config_run_parameters(
    ots_tmpdir_enter, res_scale, size_scale, pos_shift, results, surf_res
):
    # pylint: disable=too-many-arguments,too-many-locals,invalid-name
    eclcase_dir = ots_tmpdir_enter
    grid_file = os.path.join(eclcase_dir, "NORNE_ATW2013.EGRID")
    grid = Grid(grid_file, apply_mapaxes=True)

    mock_segy(grid, res_scale, size_scale, pos_shift, "norne_vol.segy")

    conf = {
        "eclbase": os.path.join(eclcase_dir, "NORNE_ATW2013"),
        "above": 100,
        "seabed": 300,
        "youngs": 0.5,
        "poisson": 0.3,
        "rfactor": 20,
        "mapaxes": False,
        "convention": 1,
        "output_dir": "ts",
        "horizon": "horizon.irap",
        "vintages_export_file": "ts.txt",
        "velocity_model": "norne_vol.segy",
        "vintages": {
            "ts_simple": [["1997-11-06", "1998-02-01"], ["1997-12-17", "1998-01-01"]],
            "dpv": [["1997-11-06", "1997-12-17"]],
        },
    }
    with open("ots_config.yml", "w", encoding="utf8") as file:
        yaml.dump(conf, file, default_flow_style=False)
    ots_run("ots_config.yml")

    # test results
    # 1. we compare the surface resolution
    # 2. we test sum of output surface heightfields for
    #   2a. top left and bottom right corners (which is due to shift segy volume)
    #   2b. we expect the surface to drop with each timeshift
    def to_numpy(surf):
        arr = surf.get_values1d(order="F").reshape(-1, surf.get_nx())
        return arr

    # horizon
    s_horizon = xtgeo.surface_from_file("horizon.irap", fformat="irap_binary")
    assert surf_res[0] == s_horizon.get_nx()
    assert surf_res[1] == s_horizon.get_ny()
    assert (s_horizon.get_nx() > grid.get_ny()) == results[-1]
    assert (s_horizon.get_ny() > grid.get_nx()) == results[-1]

    _err = 0.01
    nx = s_horizon.get_nx()
    ny = s_horizon.get_ny()
    arr = to_numpy(s_horizon)
    sh_top_left = arr[: ny // 2 - 1, : nx // 2 - 1].sum()
    sh_bottom_right = arr[ny // 2 :, nx // 2 :].sum()
    assert ((sh_top_left - _err) > 0) == results[0]
    assert ((sh_bottom_right - _err) > 0) == results[1]

    # ts_simple1
    ts_simple1 = xtgeo.surface_from_file(
        "ts_ts_simple/ots_1997_11_06_1998_02_01.irap", fformat="irap_binary"
    )
    assert nx == ts_simple1.get_nx()
    assert ny == ts_simple1.get_ny()
    assert (ts_simple1.get_nx() > grid.get_ny()) == results[-1]
    assert (ts_simple1.get_ny() > grid.get_nx()) == results[-1]

    arr = to_numpy(ts_simple1)
    ts1_top_left = arr[: ny // 2 - 1, : nx // 2 - 1].sum()
    ts1_bottom_right = arr[ny // 2 :, nx // 2 :].sum()
    assert ((ts1_top_left - _err) > 0) == results[0]
    assert ((ts1_bottom_right - _err) > 0) == results[1]
    assert ts1_top_left + ts1_bottom_right < sh_top_left + sh_bottom_right

    # ts_simple2
    ts_simple2 = xtgeo.surface_from_file(
        "ts_ts_simple/ots_1997_12_17_1998_01_01.irap", fformat="irap_binary"
    )
    assert nx == ts_simple2.get_nx()
    assert ny == ts_simple2.get_ny()
    assert (ts_simple2.get_nx() > grid.get_ny()) == results[-1]
    assert (ts_simple2.get_ny() > grid.get_nx()) == results[-1]

    arr = to_numpy(ts_simple2)
    ts2_top_left = arr[: ny // 2 - 1, : nx // 2 - 1].sum()
    ts2_bottom_right = arr[ny // 2 :, nx // 2 :].sum()
    assert ((ts2_top_left - _err) > 0) == results[0]
    assert ((ts2_bottom_right - _err) > 0) == results[1]
    assert ts2_top_left + ts2_bottom_right < sh_top_left + sh_bottom_right
    assert ts2_top_left + ts2_bottom_right < ts1_top_left + ts1_bottom_right

    # ts_dpv
    ts_dpv = xtgeo.surface_from_file(
        "ts_dpv/ots_1997_11_06_1997_12_17.irap", fformat="irap_binary"
    )
    assert nx == ts_dpv.get_nx()
    assert ny == ts_dpv.get_ny()
    assert (ts_dpv.get_nx() > grid.get_ny()) == results[-1]
    assert (ts_dpv.get_ny() > grid.get_nx()) == results[-1]

    arr = to_numpy(ts_simple2)
    dpv_top_left = arr[: ny // 2 - 1, : nx // 2 - 1].sum()
    dpv_bottom_right = arr[ny // 2 :, nx // 2 :].sum()
    assert ((dpv_top_left - _err) > 0) == results[0]
    assert ((dpv_bottom_right - _err) > 0) == results[1]
    assert dpv_top_left + dpv_bottom_right < sh_top_left + sh_bottom_right
