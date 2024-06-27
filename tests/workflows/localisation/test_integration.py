import itertools

import numpy as np
import pytest
import xtgeo
import yaml
from ert import LibresFacade
from ert.storage import open_storage
from xtgeo import RegularSurface

from semeio.workflows.localisation.local_config_script import LocalisationConfigJob

# pylint: disable=invalid-name


@pytest.mark.usefixtures("setup_poly_ert")
def test_localisation_surf():
    # pylint: disable=too-many-locals
    with open("poly.ert", "a", encoding="utf-8") as fout:
        fout.write(
            "SURFACE   PARAM_SURF_A     OUTPUT_FILE:surf.txt    "
            "INIT_FILES:surf%d.txt   BASE_SURFACE:surf0.txt"
        )
    nreal = 20
    ncol = 10
    nrow = 10
    rotation = 0.0
    xinc = 50.0
    yinc = 50.0
    xori = 0.0
    yori = 0.0
    values = np.zeros(nrow * ncol)
    for n in range(nreal):
        filename = "surf" + str(n) + ".txt"
        delta = 0.1
        for j in range(nrow):
            for i in range(ncol):
                index = i + j * ncol
                values[index] = float(j) + n * delta
        surface = RegularSurface(
            ncol=ncol,
            nrow=nrow,
            xinc=xinc,
            yinc=yinc,
            xori=xori,
            yori=yori,
            rotation=rotation,
            values=values,
        )
        surface.to_file(filename, fformat="irap_ascii")

    ert = LibresFacade.from_config_file("poly.ert")
    config = {
        "log_level": 3,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "surface_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "azimuth": 200,
                    "ref_point": [250, 250],
                    "surface_file": "surf0.txt",
                },
            },
        ],
    }

    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config, fout)
    with open_storage(ert.enspath, "w") as storage:
        ert.run_ertscript(
            LocalisationConfigJob,
            storage,
            storage.create_experiment().create_ensemble(
                name="default", ensemble_size=ert.get_ensemble_size()
            ),
            "local_config.yaml",
        )


@pytest.mark.usefixtures("setup_poly_ert")
def test_localisation_surf_const():
    # pylint: disable=too-many-locals
    with open("poly.ert", "a", encoding="utf-8") as fout:
        fout.write(
            "SURFACE   PARAM_SURF_A     OUTPUT_FILE:surf.txt    "
            "INIT_FILES:surf%d.txt   BASE_SURFACE:surf0.txt"
        )
    nreal = 20
    ncol = 10
    nrow = 10
    rotation = 0.0
    xinc = 50.0
    yinc = 50.0
    xori = 0.0
    yori = 0.0
    values = np.zeros(nrow * ncol)
    for n in range(nreal):
        filename = "surf" + str(n) + ".txt"
        delta = 0.1
        for j in range(nrow):
            for i in range(ncol):
                index = i + j * ncol
                values[index] = float(j) + n * delta
        surface = RegularSurface(
            ncol=ncol,
            nrow=nrow,
            xinc=xinc,
            yinc=yinc,
            xori=xori,
            yori=yori,
            rotation=rotation,
            values=values,
        )
        surface.to_file(filename, fformat="irap_ascii")

    ert = LibresFacade.from_config_file("poly.ert")
    config = {
        "log_level": 3,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "surface_scale": {
                    "method": "constant",
                    "value": 1.0,
                    "surface_file": "surf0.txt",
                },
            },
        ],
    }

    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config, fout)
    with open_storage(ert.enspath, "w") as storage:
        ert.run_ertscript(
            LocalisationConfigJob,
            storage,
            storage.create_experiment().create_ensemble(
                name="default", ensemble_size=ert.get_ensemble_size()
            ),
            "local_config.yaml",
        )


# This test and the test test_localisation_field2 are similar,
# but the first test a case with multiple fields and multiple
# ministeps where write_scaling_factor is activated and one
# file is written per ministep.
# Test case 2 tests four different methods for defining scaling factors for fields
@pytest.mark.usefixtures("setup_poly_ert")
def test_localisation_field1():
    # pylint: disable=too-many-locals

    # Make a 3D grid with no inactive cells
    grid_filename = "grid3D.EGRID"
    grid = create_box_grid_with_inactive_and_active_cells(
        grid_filename, has_inactive_values=False
    )

    nreal = 20
    (nx, ny, nz) = grid.dimensions
    with open("poly.ert", "a", encoding="utf-8") as fout:
        fout.write(f"GRID   {grid_filename}\n")

        property_names = ["G1", "G2", "G3", "G4", "G5", "G6"]
        for pname in property_names:
            filename_output = pname + ".roff"
            filename_input = pname + "_%d.roff"
            values = np.zeros((nx, ny, nz), dtype=np.float32)
            property_field = xtgeo.GridProperty(grid, values=0.0, name=pname)
            for n in range(nreal):
                values = np.zeros((nx, ny, nz), dtype=np.float32)
                property_field.values = values + 0.1 * n
                filename = pname + "_" + str(n) + ".roff"
                print(f"Write file: {filename}")
                property_field.to_file(filename, fformat="roff", name=pname)

            fout.write(
                f"FIELD  {pname}  PARAMETER  {filename_output}  "
                f"INIT_FILES:{filename_input}  MIN:-5.5   MAX:5.5  "
                "FORWARD_INIT:False\n"
            )

    ert = LibresFacade.from_config_file("poly.ert")
    config = {
        "log_level": 3,
        "write_scaling_factors": True,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G1", "G2"],
                },
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "azimuth": 200,
                    "ref_point": [700, 370],
                },
            },
            {
                "name": "CORR2",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G3"],
                },
                "field_scale": {
                    "method": "const_gaussian_decay",
                    "main_range": 1000,
                    "perp_range": 950,
                    "azimuth": 100,
                    "ref_point": [700, 370],
                },
            },
            {
                "name": "CORR3",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G4"],
                },
                "field_scale": {
                    "method": "const_exponential_decay",
                    "main_range": 1000,
                    "perp_range": 950,
                    "azimuth": 100,
                    "ref_point": [700, 370],
                    "normalised_tapering_range": 1.2,
                    "cutoff": True,
                },
            },
            {
                "name": "CORR4",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G5"],
                },
                "field_scale": {
                    "method": "constant",
                    "value": 1.0,
                },
            },
        ],
    }

    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config, fout)
    with open_storage(ert.enspath, "w") as storage:
        ert.run_ertscript(
            LocalisationConfigJob,
            storage,
            storage.create_experiment().create_ensemble(
                name="default", ensemble_size=ert.get_ensemble_size()
            ),
            "local_config.yaml",
        )


def create_box_grid_with_inactive_and_active_cells(
    output_grid_file, has_inactive_values=True
):
    # pylint: disable=too-many-locals
    nx = 30
    ny = 25
    nz = 3
    xinc = 50.0
    yinc = 50.0
    zinc = 10.0
    xori = 0.0
    yori = 0.0
    grid = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(xori, yori, 0.0),
        increment=(xinc, yinc, zinc),
        rotation=0.0,
        flip=1,
    )
    # Create a polygon file to use to set some grid cells inactive
    with open("polygon.txt", "w", encoding="utf-8") as fout:
        x = []
        y = []
        x.append(xori + 5 * xinc)
        y.append(yori + 5 * yinc)

        x.append(xori + (nx - 6) * xinc)
        y.append(yori + 5 * yinc)

        x.append(xori + (nx - 6) * xinc)
        y.append(yori + (ny - 6) * yinc)

        x.append(xori + 5 * xinc)
        y.append(yori + (ny - 6) * yinc)

        x.append(xori + 5 * xinc)
        y.append(yori + 5 * yinc)

        for i in range(5):
            fout.write(f" {x[i]}  {y[i]}  {zinc}\n")

    polygon = xtgeo.polygons_from_file("polygon.txt", fformat="xyz")
    if has_inactive_values:
        grid.inactivate_outside(polygon, force_close=True)

    print(f" Write file: {output_grid_file}")
    grid.to_file(output_grid_file, fformat="egrid")
    return grid


def create_region_parameter(filename, grid):
    # Create a discrete parameter to represent a region parameter
    region_param_name = "Region"
    region_code_names = {
        "RegionA": 1,
        "RegionB": 2,
        "RegionC": 3,
        "RegionD": 4,
        "RegionE": 5,
        "RegionF": 6,
    }
    region_param = xtgeo.GridProperty(
        grid, name=region_param_name, discrete=True, values=1
    )
    region_param.dtype = np.int32
    region_param.codes = region_code_names
    (nx, ny, nz) = grid.dimensions
    values = np.zeros((nx, ny, nz), dtype=np.int32)
    values[:, :, :] = 0
    for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
        if 0 <= i <= nx / 2 and 0 <= j <= ny / 2:
            if 0 <= k <= nz / 2:
                values[i, j, k] = 2
            else:
                values[i, j, k] = 5
        if nx / 2 + 1 <= i < nx and 0 <= j <= ny / 2:
            if nz / 2 <= k < nz:
                values[i, j, k] = 3
            else:
                values[i, j, k] = 4
        if ny / 3 + 1 <= j < 2 * ny / 3 and nx / 3 <= i <= nx / 2:
            if nz / 4 <= k < nz / 2:
                values[i, j, k] = 6
            else:
                values[i, j, k] = 4
    region_param.values = values
    print(f"Write file: {filename}")
    region_param.to_file(filename, fformat="grdecl", name=region_param_name)


def create_field_and_scaling_param_and_update_poly_ert(
    poly_config_file, grid_filename, grid
):
    # pylint: disable=too-many-locals,unused-argument
    (nx, ny, nz) = grid.dimensions
    property_names = ["FIELD1", "FIELD2", "FIELD3", "FIELD4", "FIELD5"]
    scaling_names = ["SCALING1", "SCALING2", "SCALING3", "SCALING4", "SCALING5"]
    nreal = 20
    nfields = len(property_names)
    with open("poly.ert", "a", encoding="utf-8") as fout:
        fout.write(f"GRID  {grid_filename}\n")
        for m in range(nfields):
            property_name = property_names[m]
            scaling_name = scaling_names[m]
            filename_output = property_name + ".roff"
            filename_input = property_name + "_%d.roff"
            scaling_filename = scaling_name + ".GRDECL"
            values = np.zeros((nx, ny, nz), dtype=np.float32)
            property_field = xtgeo.GridProperty(grid, values=0.0, name=property_name)
            scaling_field = xtgeo.GridProperty(
                grid, values=0.5 + (m - 1) * 0.2, name=scaling_name
            )
            for n in range(nreal):
                values = np.zeros((nx, ny, nz), dtype=np.float32)
                property_field.values = values + 0.1 * n
                filename = property_name + "_" + str(n) + ".roff"
                print(f"Write file: {filename}")
                property_field.to_file(filename, fformat="roff", name=property_name)
            print(f"Write file: {scaling_filename}\n")
            scaling_field.to_file(scaling_filename, fformat="grdecl", name=scaling_name)

            fout.write(
                f"FIELD  {property_name}  PARAMETER  {filename_output}  "
                f"INIT_FILES:{filename_input}  "
                "MIN:-5.5   MAX:5.5     FORWARD_INIT:False\n"
            )


@pytest.mark.usefixtures("setup_poly_ert")
def test_localisation_field2():
    # Make a 3D grid with some inactive cells
    grid_filename = "grid3D.EGRID"
    grid = create_box_grid_with_inactive_and_active_cells(grid_filename)

    # Make some field parameters and some scalingfactor parameters
    poly_config_file = "poly.ert"
    create_field_and_scaling_param_and_update_poly_ert(
        poly_config_file, grid_filename, grid
    )

    # Create a discrete parameter to represent a region parameter
    segment_filename1 = "Region1.GRDECL"
    create_region_parameter(segment_filename1, grid)
    segment_filename2 = "Region2.GRDECL"
    create_region_parameter(segment_filename2, grid)

    ert = LibresFacade.from_config_file("poly.ert")
    config = {
        "log_level": 3,
        "write_scaling_factors": True,
        "correlations": [
            {
                "name": "CORR_GAUSSIAN",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD1",
                },
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 700,
                    "perp_range": 150,
                    "azimuth": 30,
                    "ref_point": [500, 0],
                },
            },
            {
                "name": "CORR_FROM_FILE",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD2",
                },
                "field_scale": {
                    "method": "from_file",
                    "filename": "SCALING2.GRDECL",
                    "param_name": "SCALING2",
                },
            },
            {
                "name": "CORR_SEGMENT1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD3",
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": segment_filename1,
                    "param_name": "Region",
                    "active_segments": [1, 2, 4, 5],
                    "scalingfactors": [1.0, 1.5e-5, 0.3, 0.15],
                    "smooth_ranges": [2, 2],
                },
            },
            {
                "name": "CORR_SEGMENT2",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD4",
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": segment_filename1,
                    "param_name": "Region",
                    "active_segments": [1, 2, 4, 5],
                    "scalingfactors": [0.5, 1.0, 0.8, 0.05],
                    "smooth_ranges": [2, 2],
                },
            },
            {
                "name": "CORR_SEGMENT3",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD5",
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": segment_filename2,
                    "param_name": "Region",
                    "active_segments": [1, 3, 5],
                    "scalingfactors": [1.0, 0.5, 0.05],
                },
            },
        ],
    }

    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config, fout)
    with open_storage(ert.enspath, "w") as storage:
        ert.run_ertscript(
            LocalisationConfigJob,
            storage,
            storage.create_experiment().create_ensemble(
                name="default", ensemble_size=ert.get_ensemble_size()
            ),
            "local_config.yaml",
        )
