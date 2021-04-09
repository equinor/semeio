#!/usr/bin/env python

# Test example: This function is used as a forward model in ert.
# It calculates an upscaled perm value in x and y direction of 3x3x1 grid with
# permeabilities and with one multx and multy for each column and row.
# This is done for two cases (two different sets of multx, multy values but the
# same perm values.
# The measured data (results) are one value for x and one for y direction for each
# of the two cases.
# These result values are  used as predictions of observations in ERT.

import numpy as np
import yaml
from pathlib import Path

# Global variable
debug_level = 1


def read_localisation_config(specification_file_name, debug_level=0):
    if debug_level >= 1:
        print("\n")
        print(f"Localisation config file: {specification_file_name}")
    with open(specification_file_name, "r") as yml_file:
        localisation_yml = yaml.safe_load(yml_file)
    all_keywords = localisation_yml["localisation"]
    if all_keywords is None:
        raise IOError(f"None is found when reading file: {specification_file_name}")
    return all_keywords


def is_initial_iteration():
    """
    Check if folder with name equal to a non-negative integer exists.
    If a folder with name 0 is found or no folder with integer as name is found on
    top level, then the initial ensemble is used, otherwise the updated ensemble
    is used. If a folder with name 1 or 2 or 3 or ... MAXITER is found,
    the  mode is to copy/import updated GRF from ERT and this function return False
    """
    MAXITER = 100
    iterfolder = -1
    for folder in range(MAXITER):
        if Path(str(folder)).exists():
            iterfolder = folder
            break
    print(f" -- Iteration: {iterfolder}")
    return iterfolder <= 0


def get_realisation_number():
    MAXREALISATION = 500
    found = False
    for realisation in range(MAXREALISATION):
        folder = "realisation_" + str(realisation)
        realisation_folder = Path(folder)
        if realisation_folder.exists():
            real_number = realisation
            found = True
            break
    if not found:
        raise ValueError(
            f"Could not find realisation number less than {MAXREALISATION}"
        )
    print(f" -- Realisation number: {real_number}")
    return real_number


def read_mult_values(filename, number_of_values):
    # Get values for mult
    with open(filename, "r") as file:
        lines = file.readlines()
    if len(lines) != 1:
        print(f"Expecting only one line in file {filename}")
    mult = np.zeros(number_of_values, np.float32)
    if lines is not None:
        words = lines[0].split()
        for n in range(number_of_values):
            mult[n] = float(words[n])
    return mult


def read_perm(filename, nx, ny):
    with open(filename, "r") as file:
        lines = file.readlines()
    if len(lines) != ny:
        print(
            f"Number of lines in file {filename} is not consistent with "
            f"the dimensions: ({ny})"
        )
    perm = np.zeros((nx, ny), np.float32)
    if lines is not None:
        k = 0
        for m in range(ny):
            words = lines[k].split()
            perm[:, m] = np.array(list(map(float, words)))
            k = k + 1
    return perm


def arithmetic_harmonic_upscaling(perm, multx, multy, nx, ny, direction):
    if direction == "x":
        sum_inv_avg_col = 0.0
        for n in range(nx):
            sumcol = 0.0
            for m in range(ny):
                sumcol = sumcol + perm[n, m]
            avg_col = multx[n] * sumcol / ny
            sum_inv_avg_col += 1 / avg_col
        sum_inv_avg_col = sum_inv_avg_col / nx
        upscaled = 1 / sum_inv_avg_col
    else:
        sum_inv_avg_row = 0.0
        for m in range(ny):
            sumrow = 0.0
            for n in range(nx):
                sumrow = sumrow + perm[n, m]
            avg_row = multy[m] * sumrow / nx
            sum_inv_avg_row += 1 / avg_row
        sum_inv_avg_row = sum_inv_avg_row / ny
        upscaled = 1 / sum_inv_avg_row
    return upscaled


def write_result(filename, value):
    with open(filename, "w") as file:
        file.write(str(value))
    print(f"Write file: {filename}")


def run():

    nx = 3
    ny = 3

    # Get values for multx (3 values)
    param_file = "param1.dat"
    multx = read_mult_values(param_file, nx)
    print(f" -- MULTX: {multx}")

    # Get values for multy (3 values)
    param_file = "param2.dat"
    multy = read_mult_values(param_file, ny)
    print(f" -- MULTY: {multy}")

    # Get values for multx2 (3 values)
    param_file = "param1b.dat"
    multx2 = read_mult_values(param_file, nx)
    print(f" -- MULTX2: {multx2}")

    # Get values for multy2 (3 values)
    param_file = "param2b.dat"
    multy2 = read_mult_values(param_file, ny)
    print(f" -- MULTY2: {multy2}")

    # Get values for perm (3x3 values)
    param_file = "param3.dat"
    perm = read_perm(param_file, nx, ny)
    print(f" -- PERM:\n {perm[:, :]}")

    # Calculate response variables (predictions of the observations
    # ArithmeticHarmonic upscaling of 3x3 grid with perm values andmultx and multy
    upscaledx1 = arithmetic_harmonic_upscaling(perm, multx, multy, nx, ny, "x")
    upscaledx2 = arithmetic_harmonic_upscaling(perm, multx2, multy2, nx, ny, "x")
    upscaledy1 = arithmetic_harmonic_upscaling(perm, multx, multy, nx, ny, "y")
    upscaledy2 = arithmetic_harmonic_upscaling(perm, multx2, multy2, nx, ny, "y")

    print(f"Upscaled x direction for case1: {upscaledx1}")
    print(f"Upscaled y direction for case1: {upscaledy1}")
    print(f"Upscaled x direction for case2: {upscaledx2}")
    print(f"Upscaled y direction for case2: {upscaledy2}")

    # Write file with upscaled values
    resultfile = "RESULT_1_0.dat"
    write_result(resultfile, upscaledx1)

    resultfile = "RESULT_2_0.dat"
    write_result(resultfile, upscaledy1)

    resultfile = "RESULT_1b_0.dat"
    write_result(resultfile, upscaledx2)

    resultfile = "RESULT_2b_0.dat"
    write_result(resultfile, upscaledy2)


# Main
run()
