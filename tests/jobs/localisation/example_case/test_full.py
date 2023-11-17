import math
import os
import random
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import xtgeo
from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.shared.plugins.plugin_manager import ErtPluginContext
from ert.storage import open_storage
from scripts.common_functions import (
    Settings,
    generate_field_and_upscale,
    get_cell_indices,
    get_nobs,
    write_upscaled_field_to_file,
)

# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments


def generate_seed_file(
    file_name: str,
    start_seed: int = 9828862224,
    number_of_seeds: int = 1000,
):
    print(f"Generate random seed file: {file_name}")
    random.seed(start_seed)
    with open(file_name, "w", encoding="utf8") as file:
        for _ in range(number_of_seeds):
            file.write(f"{random.randint(1, 999999999)}\n")


def create_grid(
    grid_file_name: str,
    dimensions: Tuple[int, int, int],
    size: Tuple[float, float, float],
    standard_grid_index_origo: bool,
    polygon_file_name: Optional[str] = None,
):
    xsize, ysize, zsize = size
    nx, ny, nz = dimensions
    if standard_grid_index_origo:
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

    if polygon_file_name is not None and os.path.exists(polygon_file_name):
        print(f"Use polygon  file {polygon_file_name} to create actnum ")
        polygon = xtgeo.polygons_from_file(polygon_file_name, fformat="xyz")
        grid_object.inactivate_outside(polygon)

    print(f"Write grid file: {grid_file_name} ")
    grid_object.to_file(grid_file_name, fformat="egrid")
    return grid_object


def obs_positions(
    grid_dimension_upscaled: tuple,
    model_size: tuple,
    cell_indx_list: list,
    use_eclipse_origo: bool,
):
    nx, ny, _ = grid_dimension_upscaled
    xsize, ysize, _ = model_size

    dx = xsize / nx
    dy = ysize / ny

    if use_eclipse_origo:
        print("Grid index origin: Eclipse standard")
    else:
        print("Grid index origin: RMS standard")
    print(
        "Observation reference point coordinates is always "
        "from origin at lower left corner"
    )

    pos_list = []
    nobs = get_nobs(cell_indx_list)
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
    obs_index_list: list,
    field_name: str,
    corr_ranges: tuple,
    azimuth: float,
    grid_dimension: tuple,
    model_size: tuple,
    use_eclipse_grid_index_origo: bool,
    config_file_name: str = "local_config.yml",
    write_scaling: bool = True,
    localisation_method: str = "gaussian",
):
    space = " " * 2
    space2 = " " * 4
    space3 = " " * 6
    positions = obs_positions(
        grid_dimension, model_size, obs_index_list, use_eclipse_grid_index_origo
    )
    print(f"Write localisation config file: {config_file_name}")
    with open(config_file_name, "w", encoding="utf8") as file:
        file.write("log_level: 3\n")
        file.write(f"write_scaling_factors: {write_scaling}\n")
        file.write("correlations:\n")
        nobs = get_nobs(obs_index_list)
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


def write_gen_obs(
    upscaled_values,
    observation_dir: str,
    obs_file_name: str,
    obs_data_dir: str,
    cell_indx_list: list,
    rel_err: float,
    min_err: float,
):
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
        nobs = get_nobs(cell_indx_list)
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


@pytest.mark.parametrize(
    "new_settings", [pytest.param({"model_size": {"size": (100.0, 100.0, 100.0)}})]
)
def test_that_localization_works_with_different_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_settings: Dict[str, Any]
):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.update(new_settings)

    generate_seed_file(settings.field.seed_file)

    # Create grid for the field parameter
    create_grid(
        settings.field.grid_file_name,
        settings.field.grid_dimension,
        settings.model_size.size,
        settings.model_size.use_eclipse_grid_index_origo,
        settings.model_size.polygon_file,
    )

    # Create coarse grid to be used in QC of upscaled field parameter
    create_grid(
        settings.response.grid_file_name,
        settings.response.grid_dimension,
        settings.model_size.size,
        settings.model_size.use_eclipse_grid_index_origo,
        settings.model_size.polygon_file,
    )

    print("Generate field parameter and upscale this.")
    print(
        f"The upscaled field {settings.observation.reference_param_file} "
        "is used when extracting observations."
    )

    (tmp_path / "init_files").mkdir()

    # Simulate field (with trend)
    real_number = 0
    iteration = 0
    upscaled_values = generate_field_and_upscale(
        real_number,
        iteration,
        settings.field.seed_file,
        settings.field.algorithm,
        settings.field.name,
        settings.field.initial_file_name,
        settings.field.file_format,
        settings.field.grid_dimension,
        settings.model_size.size,
        settings.field.variogram,
        settings.field.correlation_range,
        settings.field.correlation_azimuth,
        settings.field.correlation_dip,
        settings.field.correlation_exponent,
        settings.field.trend_use,
        settings.field.trend_params,
        settings.field.trend_relstd,
        settings.response.name,
        settings.response.response_function,
        settings.response.upscaled_file_name,
        settings.response.grid_dimension,
        settings.response.write_upscaled_field,
        settings.model_size.use_eclipse_grid_index_origo,
    )

    # Create observations by extracting from existing upscaled field
    write_gen_obs(
        upscaled_values,
        settings.observation.directory,
        settings.observation.file_name,
        settings.observation.data_dir,
        settings.observation.selected_grid_cells,
        settings.observation.rel_error,
        settings.observation.min_abs_error,
    )

    # Write upscaled field used as reference
    # since obs are extracted from this field
    write_upscaled_field_to_file(
        upscaled_values,
        settings.observation.reference_param_file,
        upscaled_field_name=settings.observation.reference_field_name,
        selected_cell_index_list=settings.observation.selected_grid_cells,
        file_format=settings.field.file_format,
    )

    # Write file for non-adaptive localisation using distance based localisation
    write_localisation_config(
        settings.observation.selected_grid_cells,
        settings.field.name,
        settings.field.correlation_range,
        settings.field.correlation_azimuth,
        settings.response.grid_dimension,
        settings.model_size.size,
        settings.model_size.use_eclipse_grid_index_origo,
        config_file_name="local_config_gaussian_decay.yml",
        write_scaling=True,
        localisation_method="gaussian",
    )

    write_localisation_config(
        settings.observation.selected_grid_cells,
        settings.field.name,
        settings.field.correlation_range,
        settings.field.correlation_azimuth,
        settings.response.grid_dimension,
        settings.model_size.size,
        settings.model_size.use_eclipse_grid_index_origo,
        config_file_name="local_config_constant.yml",
        write_scaling=True,
        localisation_method="constant",
    )

    write_localisation_config(
        settings.observation.selected_grid_cells,
        settings.field.name,
        settings.field.correlation_range,
        settings.field.correlation_azimuth,
        settings.response.grid_dimension,
        settings.model_size.size,
        settings.model_size.use_eclipse_grid_index_origo,
        config_file_name="local_config.yml",
        write_scaling=True,
        localisation_method=settings.localisation.method,
    )

    parser = ArgumentParser(prog="test_main")

    ert_config_file = "sim_field_local.ert"
    shutil.copy(Path(__file__).parent / ert_config_file, ert_config_file)
    shutil.copy(
        Path(__file__).parent / "FieldParam_real5_iter0_A_local.grdecl",
        "FieldParam_real5_iter0_A_local.grdecl",
    )
    shutil.copy(
        Path(__file__).parent / "FieldParam_real5_iter1_A_local.grdecl",
        "FieldParam_real5_iter1_A_local.grdecl",
    )
    shutil.copy(Path(__file__).parent / "localisation.wf", "localisation.wf")
    shutil.copy(
        Path(__file__).parent / "example_test_config_A.yml", "example_test_config_A.yml"
    )
    Path("scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "scripts" / "FM_SIM_FIELD", "scripts/FM_SIM_FIELD"
    )
    shutil.copy(
        Path(__file__).parent / "scripts" / "sim_fields.py", "scripts/sim_fields.py"
    )
    shutil.copy(
        Path(__file__).parent / "scripts" / "common_functions.py",
        "scripts/common_functions.py",
    )

    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "es_prior",
            "--target-case",
            "es_posterior",
            ert_config_file,
        ],
    )

    with ErtPluginContext() as _:
        run_cli(parsed)

        facade = LibresFacade.from_config_file(ert_config_file)

        grid_file = xtgeo.grid_from_file("GRID_STANDARD.EGRID", fformat="egrid")
        es_prior_expected = xtgeo.gridproperty_from_file(
            "FieldParam_real5_iter0_A_local.grdecl",
            fformat="grdecl",
            name="FIELDPAR",
            grid=grid_file,
        )
        es_posterior_expected = xtgeo.gridproperty_from_file(
            "FieldParam_real5_iter1_A_local.grdecl",
            fformat="grdecl",
            name="FIELDPAR",
            grid=grid_file,
        )
        with open_storage(facade.enspath) as storage:
            realization = 5
            es_prior = storage.get_ensemble_by_name("es_prior")
            es_prior_results = es_prior.load_parameters("FIELDPAR").sel(
                realizations=realization
            )

            es_posterior = storage.get_ensemble_by_name("es_posterior")
            es_posterior_results = es_posterior.load_parameters("FIELDPAR").sel(
                realizations=realization
            )

        assert np.allclose(
            es_prior_expected.values3d, np.round(es_prior_results.values, 4), atol=1e-4
        )

        assert np.allclose(
            es_posterior_expected.values3d,
            np.round(es_posterior_results.values, 4),
            atol=1e-4,
        )
