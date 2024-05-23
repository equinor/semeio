from collections import namedtuple

import cwrap
import numpy as np
import pytest
from numpy import ma
from resdata.grid.rd_grid_generator import GridGenerator

from semeio.workflows.localisation.local_script_lib import (
    ConstantScalingFactor,
    ExponentialDecay,
    GaussianDecay,
    build_decay_object,
    calculate_scaling_factors_in_regions,
    smooth_parameter,
)

# pylint: disable=invalid-name

BoxDimensions = namedtuple("BoxDimensions", ["nx", "ny", "nz"])
BoxIncrements = namedtuple("BoxIncrements", ["dx", "dy", "dz"])


def create_box_grid(
    dimensions=BoxDimensions(10, 10, 3),  # noqa B008
    increments=BoxIncrements(50.0, 50.0, 10.0),  # noqa B008
    use_actnum=False,
):
    actnum = None
    if use_actnum:
        actnum = actnum_parameter(*dimensions)
    grid = GridGenerator.create_rectangular(dimensions, increments, actnum=actnum)

    return grid, *dimensions


def actnum_parameter(nx, ny, nz):
    actnum_param = np.ones((nz, ny, nx), dtype=np.int32)

    actnum_param[:, : ny // 2, :2] = 0

    actnum_param[:, -3:, -2:] = 0

    return actnum_param.ravel()


def create_region_parameter(
    grid, dimensions, scaling_per_region=None, used_regions=None
):
    # pylint: disable=too-many-locals
    # Extracting dimensions from the named tuple
    nx, ny, nz = dimensions

    # Create 3D meshgrids for indices using 'xy' indexing for Fortran order
    j, i, k = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz), indexing="xy")

    # Initialize region and scaling arrays
    region_param = ma.zeros((ny, nx, nz), dtype=np.int32)
    scaling_param = ma.zeros((ny, nx, nz), dtype=np.float32)

    # Generate the global index in Fortran order and get the active index
    global_indices = i + j * nx + k * nx * ny
    active_indices = np.vectorize(grid.get_active_index)(global_index=global_indices)

    # Set region values based on conditions
    condition_1 = (i < nx / 2) & (j < ny / 2)
    condition_2 = (i < nx / 2) & (j >= ny / 2) & (j < ny)
    condition_3 = (i >= nx / 2) & (j >= ny / 3) & (j < 2 * ny / 3)

    region_param[condition_1] = 1
    region_param[condition_2] = 2
    region_param[condition_3] = 3

    inactive = active_indices < 0
    region_param[inactive] = ma.masked

    # If scaling factors are provided
    if scaling_per_region is not None:
        scaling_param = np.take(scaling_per_region, region_param)
        scaling_param[inactive] = ma.masked

    # If used regions are specified
    if used_regions is not None:
        mask_used = np.isin(region_param, used_regions, invert=True)
        region_param[np.logical_and(mask_used, ~inactive)] = ma.masked

    return region_param.ravel(order="F"), scaling_param.ravel(order="F")


def create_parameter_from_decay_functions(method_name, grid):
    # pylint: disable=too-many-locals
    ref_pos = (250.0, 250.0)
    main_range = 150.0
    perp_range = 100.0
    azimuth = 0.0
    tapering_range = 2
    use_cutoff = True
    decay_obj = None
    constant_value = 1.0

    nx = grid.getNX()
    ny = grid.getNY()
    nz = grid.getNZ()
    filename = "tmp_" + method_name + ".grdecl"
    if method_name == "constant":
        decay_obj = ConstantScalingFactor(constant_value)
    else:
        decay_obj = build_decay_object(
            method_name,
            ref_pos,
            main_range,
            perp_range,
            azimuth,
            grid,
            tapering_range,
            use_cutoff,
        )
    data_size = nx * ny * nz
    scaling_vector = np.zeros(data_size, dtype=np.float32)
    for index in range(data_size):
        scaling_vector[index] = decay_obj(index)

    scaling_values = np.zeros(nx * ny * nz, dtype=np.float32)

    for index in range(data_size):
        global_index = grid.global_index(active_index=index)
        scaling_values[global_index] = scaling_vector[index]

    scaling_values_3d = np.reshape(scaling_values, (nx, ny, nz), "F")
    scaling_kw_name = "SCALING"
    scaling_kw = grid.create_kw(scaling_values_3d, scaling_kw_name, False)
    with cwrap.open(filename, "w") as file:
        grid.write_grdecl(scaling_kw, file)

    return scaling_values


# The selected grid cell values below should have the same value for both
# the `GaussianDecay`` and the `ExponentialDecay`` method since the
# normalized distance is 1.0 for those grid cells not at the reference point grid
# cells at (4,4,0) (4,4,1) and (4,4,2).
# Values as reference point grid cells should be 1.0.
#  index = 44   is (i,j,k) = (4,4,0)
#  index = 144  is (i,j,k) = (4,4,1)
#  index = 244  is (i,j,k) = (4,4,2)
#  index = 49   is (i,j,k) = (9,4,0)
#  index = 149  is (i,j,k) = (9,4,1)
#  index = 174  is (i,j,k) = (4,7,1)
#  index = 114  is (i,j,k) = (4,1,1)
@pytest.mark.parametrize(
    "index_list, expected",
    [
        pytest.param(
            [44, 144, 244],
            1.0,
            id=(
                "cells at (4,4,0) (4,4,1) and (4,4,2). Values "
                "as reference point grid cells should be 1.0"
            ),
        ),
        pytest.param(
            [49, 149],
            0.049787066876888275,
            id=(
                "Values at distance 5 grid cells aways (corresponding to 250 m) "
                "which is 1 range in x direction which corresponds to the "
                "perp_range since azimuth is 0."
            ),
        ),
        pytest.param(
            [114, 174],
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
    # pylint: disable=too-many-locals
    ref_pos = [225.0, 225.0]
    grid, nx, ny, nz = create_box_grid()

    main_range = 150.0
    perp_range = 250.0
    azimuth = 0.0
    use_cutoff = False
    decay_obj = method(
        ref_pos,
        main_range,
        perp_range,
        azimuth,
        grid,
        use_cutoff,
    )

    scaling_vector = np.zeros(nx * ny * nz, dtype=np.float32)
    j, i, k = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz), indexing="xy")
    global_indices = i + j * nx + k * nx * ny
    scaling_vector[global_indices] = np.vectorize(decay_obj)(global_indices)

    result = scaling_vector[index_list]
    assert (result == np.ones(len(index_list)) * expected).all()


def test_calculate_scaling_factors_in_regions(snapshot):
    """
    Test calculation of scaling factor parameter for regions.
    For visual QC of the calculated scaling factors, write the grid to EGRID format
    and the parameter to GRDECL format for easy import into a visualization tool.
    """
    grid, nx, ny, nz = create_box_grid(use_actnum=True)
    region_param_masked, _ = create_region_parameter(grid, BoxDimensions(nx, ny, nz))
    active_segment_list = [1, 2, 3]
    scaling_value_list = [1.0, 0.5, 0.2]
    smooth_range_list = None
    (
        scaling_factor_param,
        active_region_values_used_masked,
        _,
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
        BoxDimensions(nx, ny, nz),
        scaling_per_region=scaling_per_region,
        used_regions=[1, 2, 3],
    )

    smooth_range_list = [1, 1]
    smooth_param = smooth_parameter(
        grid, smooth_range_list, scaling_param, region_param_used
    )

    snapshot.assert_match(str(smooth_param), "testdata_scaling_smooth.txt")


def test_decay_function_with_new_options(snapshot):
    grid, _, _, _ = create_box_grid(
        dimensions=BoxDimensions(25, 25, 10),
        increments=BoxIncrements(20.0, 20.0, 10.0),
        use_actnum=False,
    )
    grid.save_EGRID("tmp_grid.EGRID")

    method_name = "const_gaussian_decay"
    scaling_values = create_parameter_from_decay_functions(method_name, grid)
    snapshot.assert_match(str(scaling_values), "testdata_scaling_decay_method1.txt")

    method_name = "const_exponential_decay"
    scaling_values = create_parameter_from_decay_functions(method_name, grid)
    snapshot.assert_match(str(scaling_values), "testdata_scaling_decay_method2.txt")

    method_name = "gaussian_decay"
    scaling_values = create_parameter_from_decay_functions(method_name, grid)
    snapshot.assert_match(str(scaling_values), "testdata_scaling_decay_method3.txt")

    method_name = "exponential_decay"
    scaling_values = create_parameter_from_decay_functions(method_name, grid)
    snapshot.assert_match(str(scaling_values), "testdata_scaling_decay_method4.txt")

    method_name = "constant"
    scaling_values = create_parameter_from_decay_functions(method_name, grid)
    snapshot.assert_match(str(scaling_values), "testdata_scaling_decay_method5.txt")
