import numpy as np
from numpy import ma
import itertools
import pytest
import cwrap

from semeio.workflows.localisation.local_script_lib import (
    smooth_parameter,
    calculate_scaling_factors_in_regions,
    GaussianDecay,
    ExponentialDecay,
)
from ecl.grid.ecl_grid_generator import EclGridGenerator


def create_box_grid(use_actnum=False):
    nx = 10
    ny = 10
    nz = 3
    dx = 50.0
    dy = 50.0
    dz = 10.0
    dimensions = (nx, ny, nz)
    increments = (dx, dy, dz)
    actnum = None
    if use_actnum:
        actnum = actnum_parameter(nx, ny, nz)
    grid = EclGridGenerator.create_rectangular(dimensions, increments, actnum=actnum)
    return grid, nx, ny, nz


def actnum_parameter(nx, ny, nz):
    actnum_param = np.ones(nx * ny * nz, dtype=np.int32)
    for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
        index = i + j * nx + k * nx * ny
        if 0 <= i < 2 and 0 <= j < ny / 2:
            actnum_param[index] = 0
        if nx - 2 <= i < nx and ny - 3 <= j < ny:
            actnum_param[index] = 0
    return actnum_param


def create_region_parameter(
    grid, nx, ny, nz, scaling_per_region=None, used_regions=None
):
    # pylint: disable-msg=R0912
    region_param = ma.zeros(nx * ny * nz, dtype=np.int32)
    scaling_param = ma.zeros(nx * ny * nz, dtype=np.float32)
    for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
        index = i + j * nx + k * nx * ny
        active_index = grid.get_active_index(global_index=index)
        if active_index >= 0:
            if 0 <= i < nx / 2 and 0 <= j < ny / 2:
                region_param[index] = 1
            elif 0 <= i < nx / 2 and ny / 2 <= j < ny:
                region_param[index] = 2
            elif nx / 2 <= i < nx and ny / 3 <= j < 2 * ny / 3:
                region_param[index] = 3
        else:
            region_param[index] = ma.masked

    if scaling_per_region is not None:
        for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
            index = i + j * nx + k * nx * ny
            active_index = grid.get_active_index(global_index=index)
            if active_index >= 0:
                scaling_param[index] = scaling_per_region[region_param[index]]
            else:
                scaling_param[index] = ma.masked

    if used_regions is not None:
        for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
            index = i + j * nx + k * nx * ny
            active_index = grid.get_active_index(global_index=index)
            if active_index >= 0:
                if region_param[index] not in used_regions:
                    region_param[index] = ma.masked

    return region_param, scaling_param


def write_param(filename, grid, param_values, param_name, nx, ny, nz):
    print(f"\nWrite file: {filename}")
    values = np.reshape(param_values, (nx, ny, nz), "F")
    kw = grid.create_kw(values, param_name, False)
    with cwrap.open(filename, "w") as file:
        grid.write_grdecl(kw, file)


# The selected grid cell values below should have same value for both
# the GaussianDecay and the ExponentialDecay method since the
# normalized distance is 1 for those grid cells not at the reference point grid
# cells at (4,4,0) (4,4,1) and(4,4,2). Values as reference point grid cells
# should be 1.0
#  index = 44   is (i,j,k) = (4,4,0)
#  index = 144  is (i,j,k) = (4,4,1)
#  index = 244  is (i,j,k) = (4,4,2)
#  index = 49  is (i,j,k) = (9,4,0)
#  index = 149  is (i,j,k) = (9,4,1)
#  index = 174  is (i,j,k) = (4,7,1)
#  index = 114  is (i,j,k) = (4,1,1)
@pytest.mark.parametrize(
    "index_list, expected",
    [
        pytest.param(
            (44, 144, 244),
            1.0,
            id=(
                "cells at (4,4,0) (4,4,1) and(4,4,2). Values "
                "as reference point grid cells should be 1.0"
            ),
        ),
        pytest.param(
            (49, 149),
            0.049787066876888275,
            id=(
                "Values at distance 5 grid cells aways (corresponding to 250 m) "
                "which is 1 range in x direction which corresponds to the "
                "perp_range since azimuth is 0."
            ),
        ),
        pytest.param(
            (114, 174),
            0.049787066876888275,
            id=(
                "Values at distance 3 grid cells away (corresponding to 150m) "
                "which is 1 range in y direction which corresponds to the "
                "main_range since azimuth is 0."
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        ExponentialDecay,
        GaussianDecay,
    ],
)
def test_exponentialtype_decay_functions(method, index_list, expected):
    ref_pos = [225.0, 225.0]
    grid, nx, ny, nz = create_box_grid()

    main_range = 150.0
    perp_range = 250.0
    azimuth = 0.0
    decay_obj = method(ref_pos, main_range, perp_range, azimuth, grid)

    data_size = nx * ny * nz
    scaling_vector = np.zeros(data_size, dtype=np.float32)
    for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
        index = i + j * nx + k * nx * ny
        scaling_vector[index] = decay_obj(index)

    result = [scaling_vector[i] for i in index_list]
    assert result == [expected] * len(index_list)


def test_calculate_scaling_factors_in_regions(snapshot):
    """
    Test calculation of scaling factor parameter for regions.
    For visual QC of the calculated scaling factors, write the grid to EGRID format
    and the parameter to GRDECL format for easy import into visualization tool.
    Add e.g the following code to the end of this function with  som chosen filename
    and param_name and param_values:
          grid_filename = "box_grid.EGRID"
          grid.save_EGRID(grid_filename)
          param_filename = "param.grdecl"
          param_name = "param"
          write_param(param_filename, grid, param_values, param_name, nx, ny, nz)
    """
    grid, nx, ny, nz = create_box_grid(use_actnum=True)
    region_param_masked, _ = create_region_parameter(grid, nx, ny, nz)
    active_segment_list = [1, 2, 3]
    scaling_value_list = [1.0, 0.5, 0.2]
    smooth_range_list = None
    (
        scaling_factor_param,
        active_region_values_used_masked,
        regions_in_param,
    ) = calculate_scaling_factors_in_regions(
        grid,
        region_param_masked,
        active_segment_list,
        scaling_value_list,
        smooth_range_list,
    )
    active_region_param = np.zeros(nx * ny * nz, dtype=np.int32)
    active = ~active_region_values_used_masked.mask
    active_region_param[active] = active_region_values_used_masked[active]

    region_param = np.zeros(nx * ny * nz, dtype=np.int32)
    active_specified = ~region_param_masked.mask
    region_param[active_specified] = region_param_masked[active_specified]

    snapshot.assert_match(str(scaling_factor_param), "testdata_scaling.txt")


def test_smooth_parameter(snapshot):
    grid, nx, ny, nz = create_box_grid(use_actnum=True)

    scaling_per_region = [0, 1.0, 0.5, 0.2]
    region_param_used, scaling_param = create_region_parameter(
        grid,
        nx,
        ny,
        nz,
        scaling_per_region=scaling_per_region,
        used_regions=[1, 2, 3],
    )

    smooth_range_list = [1, 1]
    smooth_param = smooth_parameter(
        grid, smooth_range_list, scaling_param, region_param_used
    )

    snapshot.assert_match(str(smooth_param), "testdata_scaling_smooth.txt")
