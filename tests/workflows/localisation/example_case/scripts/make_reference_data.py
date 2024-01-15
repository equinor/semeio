#!/usr/bin/env python
"""
This script can be used to create reference data for the pytest  test_full.py
It uses the same configurations, the same code and random seed and run ERT
for each case using localisation. Note that each time the reference data
(GRDECL files ofr one selected realization) need to be updated, they must
be quality checked that they are correct. Visualize them , compare runs
with and without localisation to check the differences.
"""
# pylint: disable=import-error,missing-function-docstring,redefined-outer-name

import copy
import subprocess
import yaml
from common_functions import (
    Settings,
    initialize_case,
    example_cases,
)


def example_config(case_name, use_localisation):
    settings = Settings()
    settings.reset_to_default()
    new_settings = example_cases(case_name)
    settings.update(new_settings)

    updated_settings = copy.deepcopy(settings)
    updated_settings.localisation.use_localisation = use_localisation

    # Write yml file for the sim_fields.py script
    main_settings_dict = updated_settings.to_dict()
    with open("example_config.yml", "w", encoding="utf-8") as file:
        file.write(
            yaml.safe_dump(main_settings_dict, indent=4, default_flow_style=False)
        )
    return updated_settings


def run_case(settings, select_real=5, select_iter=1):
    initialize_case(settings)
    # Run case in ERT
    print("Run ERT")
    case_name = settings.case_name
    use_localisation = settings.localisation.use_localisation
    local = "_local" if use_localisation else ""
    command = [
        "ert",
        "ensemble_smoother",
        "--target-case",
        f"case_{case_name}{local}",
        "sim_field.ert",
    ]
    subprocess.run(command, check=True)

    grid_file_name = settings.field.grid_file_name
    file_format = settings.field.file_format.lower()
    # Select one of the realizations for the initial and updated
    # ensemble as a reference for each case.
    # Convert the files to GRDECL format. If the cases have
    # generated correct results, save the references
    # to git as reference data for the test_full.py  pytest script.
    print("Convert reference file to GRDECL format")
    command = [
        "cp",
        f"simulations/sim_field_{case_name}{local}/"
        f"realization-{select_real}/iter-0/init_files/FieldParam.{file_format}",
        f"init_files/FieldParam_real{select_real}_iter0_{case_name}{local}."
        f"{file_format}",
    ]
    subprocess.run(command, check=True)

    command = [
        "cp",
        f"simulations/sim_field_{case_name}{local}/"
        f"realization-{select_real}/iter-{select_iter}/FieldParam.{file_format}",
        f"init_files/FieldParam_real{select_real}_iter{select_iter}_{case_name}{local}."
        f"{file_format}",
    ]
    subprocess.run(command, check=True)

    command = [
        "./scripts/roff_to_grdecl.py",
        f"init_files/FieldParam_real{select_real}_iter0_{case_name}{local}."
        f"{file_format}",
        f"init_files/FieldParam_real{select_real}_iter0_{case_name}{local}.grdecl",
        f"{grid_file_name}",
    ]
    subprocess.run(command, check=True)

    command = [
        "./scripts/roff_to_grdecl.py",
        f"init_files/FieldParam_real{select_real}_iter{select_iter}_{case_name}{local}."
        f"{file_format}",
        f"init_files/FieldParam_real{select_real}_iter{select_iter}_{case_name}{local}."
        "grdecl",
        f"{grid_file_name}",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    # Realisation number selected to be used as reference realisation
    SELECT_REAL = 5
    # Use ensemble smoother
    SELECT_ITER = 1
    for case_name in ["A", "B", "C", "D", "E", "F", "G", "A2"]:
        print(f"Case: {case_name}")
        # Make the file example_config.yml to be used in the ERT
        # forward model sim_field.py
        settings_with_localisation = example_config(case_name, use_localisation=True)
        settings_without_localisation = example_config(
            case_name, use_localisation=False
        )

        # Initialize the case:
        # - create seed file,
        # - create grids,
        # - simulate one unconditioned realization,
        # - calculate response,
        # - extract synthetic observations from response,
        # - create localisation config file,  (local_config.yml)
        # - create ert config file (sim_field.ert)
        # - prepare region parameter for case using regions

        print("Run without localisation")
        run_case(
            settings_without_localisation,
            select_real=SELECT_REAL,
            select_iter=SELECT_ITER,
        )

        print("\nRun with localisation")
        run_case(
            settings_with_localisation,
            select_real=SELECT_REAL,
            select_iter=SELECT_ITER,
        )
