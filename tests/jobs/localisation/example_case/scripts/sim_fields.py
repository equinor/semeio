#!/usr/bin/env python
"""
Script used as forward model in ERT to test localisation.
"""
import sys

# pylint: disable=import-error, redefined-outer-name
# pylint: disable=missing-function-docstring,invalid-name
from common_functions import (
    generate_field_and_upscale,
    get_cell_indices,
    get_nobs,
    read_config_file,
    read_field_from_file,
    read_obs_field_from_file,
    read_upscaled_field_from_file,
    settings,
    upscaling,
    write_obs_pred_diff_field,
)


def write_prediction_gen_data(upscaled_values):
    """
    Write GEN_DATA file with predicted values of observables (selected upscaled values)
    """
    cell_indx_list = settings.observation.selected_grid_cells
    response_file_name = settings.response.gen_data_file_name
    print(f"Write GEN_DATA file with prediction of observations: {response_file_name}")
    with open(response_file_name, "w", encoding="utf8") as file:
        # NOTE: The sequence of values must be the same as for the observations
        nobs = get_nobs()
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, nobs, cell_indx_list)
            value = upscaled_values[Iindx, Jindx, Kindx]
            print(f"Prediction of obs for {Iindx+1},{Jindx+1},{Kindx+1}: {value}")
            file.write(f"{value}\n")


def get_iteration_real_number_config_file(argv):
    if len(argv) < 4:
        raise IOError(
            "Missing command line arguments <iteration> <real_number> <config_file>"
        )
    arg1 = argv[1]
    if arg1 is None:
        raise IOError(
            "Missing iteration number (argv[1]) when running this script manually"
        )
    iteration = int(arg1)
    print(f"ERT iteration: {iteration}")

    arg2 = argv[2]
    if arg2 is None:
        raise IOError("Missing real_number (argv[2]) when running this script manually")
    real_number = int(arg2)
    print(f"ERT realization: {real_number}")

    config_file_name = argv[3]
    return iteration, real_number, config_file_name


def main(args):
    """
    For iteration = 0:
    - simulate field, export to file as initial ensemble realization
    - upscale and extract predicted values for observables
      (selected coarse grid cell values)
    For iteration > 0:
    - Import updated field from ERT.
    - upscale and extract predicted values for observables
      (selected coarse grid cell values)
    """

    # NOTE: Both the fine scale grid with simulated field values
    #  and the coarse grid with upscaled values must have Eclipse grid index origin

    # Read config_file if it exists. Use default settings for everything not specified.
    iteration, real_number, config_file_name = get_iteration_real_number_config_file(
        args
    )
    read_config_file(config_file_name)

    if iteration == 0:
        print(f"Generate new field parameter realization:{real_number} ")
        # Simulate field (with trend)
        upscaled_values = generate_field_and_upscale(real_number)

    else:
        print(f"Import updated field parameter realization: {real_number} ")
        field_object = read_field_from_file()
        field_values = field_object.values

        # Calculate upscaled values for selected coarse grid cells
        upscaled_values = upscaling(
            field_values,
            iteration=iteration,
        )
    # Write GEN_DATA file
    write_prediction_gen_data(upscaled_values)

    # Optional output calculate difference between upscaled field and
    # and reference upscaled field
    if settings.optional.write_obs_pred_diff_field_file:
        obs_field_object = read_obs_field_from_file()
        upscaled_field_object = read_upscaled_field_from_file(iteration)
        write_obs_pred_diff_field(upscaled_field_object, obs_field_object)


if __name__ == "__main__":
    # Command line arguments are iteration real_number  test_case_config_file
    main(sys.argv)
