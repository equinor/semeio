#!/usr/bin/env python

# Read ROFF file with GRF values. Depending on iteration (initial ensemble or update)
# it will read the GRF file from two different directories.
# Pick the cell value (5,5,3) (counting from 1) and write it to the result file. This value is the prediction of the observation for the realisation
#Copy the GRF.roff file 
import numpy as np
import yaml
from pathlib import Path

# Global variable
debug_level = 1

def read_localisation_config(specification_file_name, debug_level=0):
    if debug_level >= 1:
        print('\n')
        print(f'Localisation config file: {specification_file_name}')
    with open(specification_file_name, 'r') as yml_file:
        localisation_yml = yaml.safe_load(yml_file)
    all_keywords = localisation_yml['localisation']
    if all_keywords is None:
        raise IOError(f'None is found when reading file: {specification_file_name}')
    return all_keywords


def is_initial_iteration():
    '''
    Check if folder with name equal to a non-negative integer exists.
    If a folder with name 0 is found or no folder with integer as name is found on top level, 
   then the initial ensemble is used, otherwise the updated ensemble is used.
    If a folder with name 1 or 2 or 3 or ... MAXITER is found,
    the  mode is to copy/import updated GRF from ERT and this function return False
    '''
    MAXITER = 100
    iterfolder = -1
    for folder in range(MAXITER):
        if Path(str(folder)).exists():
            iterfolder = folder
            break
    print(f' -- Iteration: {iterfolder}')
    return iterfolder <= 0

def get_realisation_number():
    MAXREALISATION = 500
    found = False
    for realisation in range(MAXREALISATION):
        folder = 'realisation_' + str(realisation)
        realisation_folder =  Path(folder)
        if realisation_folder.exists():
           real_number = realisation
           found = True
           break
    if not found:
           raise ValueError(f'Could not find realisation number less than {MAXREALISATION}')
    print(f' -- Realisation number: {real_number}')
    return real_number



def run():

    nx = 3
    ny = 3
    perm = np.zeros((nx,ny), np.float32)
    multx = np.zeros(nx, np.float32)
    multy = np.zeros(ny, np.float32)

    #Get values for 3x3 grid cell values
    param_file = 'param.dat'
    with open(param_file, "r") as file:
        lines =file.readlines()
    if len(lines) != (nx + ny + nx*ny):
        print(f'Number of lines in file {param_file} is not consistent with the dimensions: ({nx}, {ny})')
    if lines is not None:
        words =  lines[0].split()
        for n in range(nx):
            multx[n] = float(words[n])
        print(f' -- MULTX: {multx}')

        words =  lines[1].split()
        for n in range(ny):
            multy[n] = float(words[n])
        print(f' -- MULTY: {multy}')

        k = 2
        for m in range(ny):
            for n in range(nx):
                words = lines[k].split()
            perm[: , m] = np.array(list(map(float, lines[k].split())))
            k = k +1
        print(f' -- PERM:\n {perm[:, :]}')

    # Calculate response variables (predictions of the observations
    # ArithmeticHarmonic upscaling of 3x3 grid with perm values andmultx and multy
    sum_avg_col = 0.0
    sum_inv_avg_col = 0.0
    for n in range(3):
        sumcol = 0.0
        for m in range(3):
            sumcol = sumcol + perm[n,m]
        avg_col =  multx[n] * sumcol / 3.0
        sum_inv_avg_col  += 1 / avg_col
    sum_inv_avg_col  = sum_inv_avg_col / 3
    upscaled1 = 1 / sum_inv_avg_col

    sum_avg_row = 0.0
    sum_inv_avg_row  =0.0
    for m in range(3):
        sumrow = 0.0
        for n in range(3):
            sumrow = sumrow + perm[n,m]
        avg_row =  multy[m] * sumrow / 3.0
        sum_inv_avg_row  += 1 / avg_row
    sum_inv_avg_row  = sum_inv_avg_row / 3
    upscaled2 = 1 / sum_inv_avg_row
    print(f'Upscaled x direction: {upscaled1}')
    print(f'Upscaled y direction: {upscaled2}')

    # Write file with upscaled values
    resultfile1 = 'RESULT_1_0.dat'
    with open(resultfile1,'w') as file:
        file.write(str(upscaled1) )
        print(f'Write file: {resultfile1}')

    resultfile2 = 'RESULT_2_0.dat'
    with open(resultfile2,'w') as file:
        file.write(str(upscaled2) )
        print(f'Write file: {resultfile2}')

# Main
run()



