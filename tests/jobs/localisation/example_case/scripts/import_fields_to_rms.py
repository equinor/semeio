"""
Import field parameters into RMS (Must be included as python job
in RMS workflow and edited to fit your scratch directory)
Variable project is defined when running within RMS,
but not outside since it refers to Roxar API.
"""

from pathlib import Path

import xtgeo


# pylint: disable=bare-except, too-many-arguments, too-many-branches
# pylint: disable=too-many-locals, too-many-statements
def import_from_scratch_directory(
    project,
    grid_model_name,
    field_names,
    case_name,
    scratch_dir,
    file_format,
    max_iteration,
):
    """
    Import files with initial ensemble and updated fields into RMS project
    """
    grid = xtgeo.grid_from_roxar(project, grid_model_name, project.current_realisation)

    path = Path(scratch_dir)
    if not path.exists():
        raise IOError(f"File path: {scratch_dir} does not exist. ")

    real = project.current_realisation
    print("\n")
    print(f"Realization: {real} ")
    for name in field_names:
        for iteration in [0, max_iteration]:
            print(f"Iteration: {iteration}")
            if iteration == 0:
                name_with_iter = name + "_" + case_name + "_" + str(iteration)
                path = (
                    scratch_dir
                    + "realization-"
                    + str(real)
                    + "/iter-"
                    + str(iteration)
                    + "/init_files/"
                )
                if file_format == "ROFF":
                    file_name = path + name + ".roff"
                elif file_format == "GRDECL":
                    file_name = path + name + ".GRDECL"
                else:
                    raise IOError(f"Unknown file format: {file_format} ")
                print(f"File name: {file_name}  ")

                try:
                    if file_format == "ROFF":
                        # print(f"Import ROFF file: {file_name} ")
                        property0 = xtgeo.gridproperty_from_file(
                            file_name, fformat="roff"
                        )
                    else:
                        # print(f"Import GRDECL file: {file_name} ")
                        property0 = xtgeo.gridproperty_from_file(
                            file_name, fformat="grdecl", name="FIELDPAR", grid=grid
                        )
                    print(
                        f"Import property {property0.name} for iteration {iteration} "
                        f"into {name_with_iter}  "
                    )
                    property0.to_roxar(
                        project, grid_model_name, name_with_iter, realisation=real
                    )
                except:  # noqa: E722
                    print(f"Skip realization: {real} for iteration: {iteration}  ")
            elif iteration == max_iteration:
                name_with_iter = name + "_" + case_name + "_" + str(iteration)
                path = (
                    scratch_dir
                    + "realization-"
                    + str(real)
                    + "/iter-"
                    + str(iteration)
                    + "/"
                )
                if file_format == "ROFF":
                    file_name = path + name + ".roff"
                elif file_format == "GRDECL":
                    file_name = path + name + ".GRDECL"
                print(f"File name: {file_name}  ")

                try:
                    if file_format == "ROFF":
                        property3 = xtgeo.gridproperty_from_file(file_name, "roff")
                    else:
                        property3 = xtgeo.gridproperty_from_file(
                            file_name, fformat="grdecl", name="FIELDPAR", grid=grid
                        )
                    print(
                        f"Import property {property3.name} for iteration {iteration} "
                        f"into {name_with_iter}  "
                    )
                    property3.to_roxar(
                        project, grid_model_name, name_with_iter, realisation=real
                    )
                except:  # noqa: E722
                    print(f"Skip realization: {real} for iteration: {iteration}  ")
                try:
                    diff_property = property0
                    diff_property.values = property3.values - property0.values
                    name_diff = name + "_" + case_name + "_diff"
                    print(
                        f"Calculate difference between iteration {max_iteration} "
                        f" and 0:  {name_diff}"
                    )
                    diff_property.to_roxar(
                        project, grid_model_name, name_diff, realisation=real
                    )
                except:  # noqa: E722
                    print(f"Skip difference for realisation: {real} ")
