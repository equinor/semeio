#!/usr/bin/env python
"""
Script initialize the test case by creating the grid files, observation files etc
"""

import math
import os
import random
import sys

# pylint: disable=import-error
import xtgeo
from common_functions import (
    generate_field_and_upscale,
    get_cell_indices,
    get_nobs,
    read_config_file,
    settings,
    write_upscaled_field_to_file,
)

# pylint: disable=too-many-arguments,invalid-name,missing-function-docstring
# pylint: disable=too-many-locals,redefined-outer-name


def generate_seed_file(
    start_seed: int = 9828862224,
    number_of_seeds: int = 1000,
):
    # pylint: disable=unused-variable

    seed_file_name = settings.field.seed_file
    print(f"Generate random seed file: {seed_file_name}")
    random.seed(start_seed)
    with open(seed_file_name, "w", encoding="utf8") as file:
        for i in range(number_of_seeds):
            file.write(f"{random.randint(1, 999999999)}\n")


def obs_positions():
    NX, NY, _ = settings.response.grid_dimension
    use_eclipse_origo = settings.grid_size.use_eclipse_grid_index_origo

    xsize = settings.grid_size.xsize
    ysize = settings.grid_size.ysize
    dx = xsize / NX
    dy = ysize / NY
    cell_indx_list = settings.observation.selected_grid_cells
    if use_eclipse_origo:
        print("Grid index origin: Eclipse standard")
    else:
        print("Grid index origin: RMS standard")
    print(
        "Observation reference point coordinates is always "
        "from origin at lower left corner"
    )

    pos_list = []
    nobs = get_nobs()
    for obs_number in range(nobs):
        (Iindx, Jindx, _) = get_cell_indices(obs_number, nobs, cell_indx_list)
        x = (Iindx + 0.5) * dx
        if use_eclipse_origo:
            y = ysize - (Jindx + 0.5) * dy
        else:
            y = (Jindx + 0.5) * dy

        pos_list.append((x, y))

    return pos_list


def write_localisation_config(
    config_file_name="local_config.yml",
    write_scaling=True,
    localisation_method="gaussian",
):
    obs_index_list = settings.observation.selected_grid_cells
    field_name = settings.field.name
    corr_ranges = settings.field.correlation_range
    azimuth = settings.field.correlation_azimuth
    space = " " * 2
    space2 = " " * 4
    space3 = " " * 6
    positions = obs_positions()
    print(f"Write localisation config file: {config_file_name}")
    with open(config_file_name, "w", encoding="utf8") as file:
        file.write("log_level: 3\n")
        file.write(f"write_scaling_factors: {write_scaling}\n")
        file.write("correlations:\n")
        nobs = get_nobs()
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, nobs, obs_index_list)
            obs_name = f"OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}"
            pos = positions[obs_number]
            file.write(f"{space}- name: CORR_{obs_number}\n")
            file.write(f"{space2}obs_group:\n")
            file.write(f'{space3}add: ["{obs_name}"]\n')
            file.write(f"{space2}param_group:\n")
            file.write(f'{space3}add: ["{field_name}"]\n')
            file.write(f"{space2}field_scale:\n")
            if localisation_method == "gaussian":
                file.write(f"{space3}method: gaussian_decay\n")
                file.write(f"{space3}main_range: {corr_ranges[0]}\n")
                file.write(f"{space3}perp_range: {corr_ranges[1]}\n")
                file.write(f"{space3}azimuth: {azimuth}\n")
                file.write(f"{space3}ref_point: [ {pos[0]}, {pos[1]} ]\n")
            elif localisation_method == "constant":
                file.write(f"{space3}method: constant\n")
                file.write(f"{space3}value: 1.0\n")


def write_gen_obs(upscaled_values):
    observation_dir = settings.observation.directory
    obs_file_name = settings.observation.file_name
    obs_data_dir = settings.observation.data_dir
    cell_indx_list = settings.observation.selected_grid_cells
    rel_err = settings.observation.rel_error
    min_err = settings.observation.min_abs_error
    if not os.path.exists(observation_dir):
        print(f"Create directory: {observation_dir} ")
        os.makedirs(observation_dir)
    data_dir = observation_dir + "/" + obs_data_dir
    if not os.path.exists(data_dir):
        print(f"Create directory: {data_dir} ")
        os.makedirs(data_dir)

    print(f"Write observation file: {obs_file_name} ")
    filename = observation_dir + "/" + obs_file_name
    with open(filename, "w", encoding="utf8") as obs_file:
        # Check if cell_indx_list is a single tuple (i,j,k)
        # or a list of tuples  of type (i,j,k)
        nobs = get_nobs()
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, nobs, cell_indx_list)
            value = upscaled_values[Iindx, Jindx, Kindx]
            value_err = math.fabs(value) * rel_err
            value_err = max(value_err, min_err)

            obs_data_relative_file_name = (
                obs_data_dir
                + "/obs_"
                + str(Iindx + 1)
                + "_"
                + str(Jindx + 1)
                + "_"
                + str(Kindx + 1)
                + ".txt"
            )

            obs_file.write(f"GENERAL_OBSERVATION   OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}  ")
            obs_file.write("{ ")
            obs_file.write(
                f"DATA = RESULT_UPSCALED_FIELD ; INDEX_LIST = {obs_number} ; "
            )
            obs_file.write("RESTART = 0;  ")
            obs_file.write(f"OBS_FILE = ./{obs_data_relative_file_name} ; ")
            obs_file.write(" };\n")

            data_file_name = observation_dir + "/" + obs_data_relative_file_name
            print(f"Write file: {data_file_name} ")
            with open(data_file_name, "w", encoding="utf8") as data_file:
                data_file.write(f"{value}  {value_err}\n")


def create_grid():
    grid_file_name = settings.field.grid_file_name
    nx, ny, nz = settings.field.grid_dimension
    xsize = settings.grid_size.xsize
    ysize = settings.grid_size.ysize
    zsize = settings.grid_size.zsize
    if settings.grid_size.use_eclipse_grid_index_origo:
        flip = -1
        x0 = 0.0
        y0 = ysize
        z0 = 0.0
    else:
        flip = 1
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    grid_object = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(x0, y0, z0),
        increment=(dx, dy, dz),
        rotation=0.0,
        flip=flip,
    )

    polygon_file_name = settings.grid_size.polygon_file
    if polygon_file_name is not None and os.path.exists(polygon_file_name):
        print(f"Use polygon  file {polygon_file_name} to create actnum ")
        polygon = xtgeo.polygons_from_file(polygon_file_name, fformat="xyz")
        grid_object.inactivate_outside(polygon)

    print(f"Write grid file: {grid_file_name} ")
    grid_object.to_file(grid_file_name, fformat="egrid")
    return grid_object


def create_upscaled_grid():
    grid_file_name = settings.response.grid_file_name
    nx, ny, nz = settings.response.grid_dimension
    xsize = settings.grid_size.xsize
    ysize = settings.grid_size.ysize
    zsize = settings.grid_size.zsize
    if settings.grid_size.use_eclipse_grid_index_origo:
        flip = -1
        x0 = 0.0
        y0 = ysize
        z0 = 0.0
    else:
        flip = 1
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    grid_object = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(x0, y0, z0),
        increment=(dx, dy, dz),
        rotation=0.0,
        flip=flip,
    )

    print(f"Write grid file: {grid_file_name} ")
    grid_object.to_file(grid_file_name, fformat="egrid")
    return grid_object


def main(config_file_name=None):
    """
    Initialize seed file, grid files, observation files and localisation config file
    """

    read_config_file(config_file_name)

    # Create seed file
    generate_seed_file()

    # Create grid for the field parameter
    create_grid()

    # Create coarse grid to be used in QC of upscaled field parameter
    create_upscaled_grid()

    print("Generate field parameter and upscale this.")
    print(
        f"The upscaled field {settings.observation.reference_param_file} "
        "is used when extracting observations."
    )

    # Simulate field (with trend)
    real_number = 0
    upscaled_values = generate_field_and_upscale(real_number)

    # Create observations by extracting from existing upscaled field
    write_gen_obs(upscaled_values)

    # Write upscaled field used as reference
    # since obs are extracted from this field
    write_upscaled_field_to_file(
        upscaled_values,
        settings.observation.reference_param_file,
        selected_cell_index_list=settings.observation.selected_grid_cells,
        file_format=settings.field.file_format,
        upscaled_field_name=settings.observation.reference_field_name,
    )

    # Write file for non-adaptive localisation using distance based localisation
    write_localisation_config(
        config_file_name="local_config_gaussian_decay.yml",
        write_scaling=True,
        localisation_method="gaussian",
    )
    write_localisation_config(
        config_file_name="local_config_constant.yml",
        write_scaling=True,
        localisation_method="constant",
    )
    write_localisation_config(
        config_file_name="local_config.yml",
        write_scaling=True,
        localisation_method=settings.localisation.method,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IOError(
            f"The script {sys.argv[0]} requires one input with test config yml file"
        )

    config_file_name = sys.argv[1]
    main(config_file_name)
