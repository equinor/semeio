"""
Common functions used by the scripts: init_test_case.py and sim_field.py
"""
import copy
import dataclasses
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Tuple

import gstools as gs
import numpy as np
import xtgeo
import yaml

# pylint: disable=too-many-lines
# pylint: disable=missing-function-docstring, too-many-locals, invalid-name
# pylint: disable=raise-missing-from
# pylint: disable=too-many-nested-blocks


# Settings for the test case in the following dataclasses
@dataclass
class ModelSize:
    """
    Length, width, thickness of a box containing the field
    Same size is used for both fine scale grid with
    the simulated field and the coarse scale grid
    containing upscaled values of the simulated field.
    """

    size: Tuple[float] = (1000.0, 2000.0, 50.0)
    polygon_file: str = None
    use_eclipse_grid_index_origo: bool = True

    def reset_to_default(self):
        self.size = (1000.0, 2000.0, 50.0)
        self.polygon_file = None
        self.use_eclipse_grid_index_origo = True


@dataclass
class Field:
    """
    Define the dimension (number of grid cells) for fine scale grid,
    name of output files and specification of model parameters for
    simulation of gaussian field with option to use linear trend.
    Relative standard deviation specify standard deviation of
    gaussian residual field relative to the trends span of value
    (max trend value - min trend value)
    """

    # pylint: disable=too-many-instance-attributes
    name: str = "FIELDPAR"
    algorithm: str = "gstools"
    file_format: str = "ROFF"
    initial_file_name_prefix: str = "init_files/FieldParam"
    updated_file_name_prefix: str = "FieldParam"
    seed_file: str = "randomseeds.txt"
    variogram: str = "gaussian"
    correlation_range: Tuple[float] = (250.0, 500.0, 2.0)
    correlation_azimuth: float = 0.0
    correlation_dip: float = 0.0
    correlation_exponent: float = 1.9
    trend_use: bool = False
    trend_params: Tuple[float] = (1.0, -1.0)
    trend_relstd: float = 0.15
    grid_dimension: Tuple[int] = (10, 20, 1)
    grid_file_name: str = "GRID_STANDARD.EGRID"

    def reset_to_default(self):
        self.name = "FIELDPAR"
        self.algorithm = "gstools"
        self.file_format = "ROFF"
        self.initial_file_name_prefix = "init_files/FieldParam"
        self.updated_file_name_prefix = "FieldParam"
        self.seed_file = "randomseeds.txt"
        self.variogram = "gaussian"
        self.correlation_range = (250.0, 500.0, 2.0)
        self.correlation_azimuth = 0.0
        self.correlation_dip = 0.0
        self.correlation_exponent = 1.9
        self.trend_use = False
        self.trend_params = (1.0, -1.0)
        self.trend_relstd = 0.15
        self.grid_dimension = (10, 20, 1)
        self.grid_file_name = "GRID_STANDARD.EGRID"


@dataclass
class Response:
    """
    Specify the coarse grid dimensions, name of file and type
    of average operation to calculated upscaled values that
    are predictions of observations of the same grid cells.
    Which cell indices are observed are specified in
    observation settings.
    """

    # pylint: disable=too-many-instance-attributes
    name: str = "UPSCALED"
    grid_dimension: Tuple[int] = (2, 4, 1)
    upscaled_file_name: str = "Upscaled"
    grid_file_name: str = "GRID_STANDARD_UPSCALED.EGRID"
    file_format: str = "ROFF"
    write_upscaled_field: bool = True
    response_function: str = "average"
    gen_data_file_prefix: str = "UpscaledField"
    ert_config_template_file: str = "init_files/sim_field_template.ert"
    ert_config_file: str = "sim_field.ert"

    def reset_to_default(self):
        self.name = "UPSCALED"
        self.grid_dimension = (2, 4, 1)
        self.upscaled_file_name = "Upscaled"
        self.grid_file_name = "GRID_STANDARD_UPSCALED.EGRID"
        self.file_format = "ROFF"
        self.write_upscaled_field = True
        self.response_function = "average"
        self.gen_data_file_prefix = "UpscaledField"
        self.ert_config_template_file = "init_files/sim_field_template.ert"
        self.ert_config_file = "sim_field.ert"


# pylint: disable=too-many-instance-attributes
@dataclass
class Observation:
    """
    Specify name of files for generated observations
    and also position of observations. Grid cell indices for grid cells
    with observations are calculated from position of observations.
    """

    directory: str = "observations"
    file_name: str = "observations.obs"
    data_dir: str = "obs_data"
    reference_param_file: str = "init_files/ObsField"
    reference_field_name: str = "ObsField"
    rel_error: float = 0.10
    min_abs_error: float = 0.01
    selected_grid_cells: Tuple[Tuple[int]] = None
    obs_positions: Tuple[Tuple[float]] = None
    obs_values: Tuple[float] = None

    def reset_to_default(self):
        self.directory = "observations"
        self.file_name = "observations.obs"
        self.data_dir = "obs_data"
        self.reference_param_file = "init_files/ObsField"
        self.reference_field_name = "ObsField"
        self.rel_error = 0.10
        self.min_abs_error = 0.01
        self.selected_grid_cells = None
        self.obs_positions = None
        self.obs_values = None


@dataclass
class Localisation:
    """
    Specify settings for the localisation config file.
    """

    method: str = "gaussian"
    region_polygons: str = None
    region_file: str = None
    scaling_file: str = None
    use_localisation: bool = True

    def reset_to_default(self):
        self.method = "gaussian"
        self.region_polygons = None
        self.region_file = None
        self.scaling_file = None
        self.use_localisation = True


@dataclass
class Optional:
    """
    Specify if some optional files should be
    written or not (for QC purpose).
    """

    write_obs_pred_diff_field_file: bool = False

    def reset_to_default(self):
        self.write_obs_pred_diff_field_file = False


@dataclass
class Settings:
    """
    Settings for the test case
    """

    case_name: str = None
    model_size: ModelSize = None
    field: Field = None
    response: Response = None
    observation: Observation = None
    localisation: Localisation = None
    optional: Optional = None

    def update(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if dataclasses.is_dataclass(attr):
                    for attr_key, attr_value in value.items():
                        if hasattr(attr, attr_key):
                            setattr(attr, attr_key, attr_value)
                else:
                    setattr(self, key, value)
        self.set_observed_cell_indices()

    def to_dict(self):
        main_dict = {"settings": {}}
        for key, value in asdict(self).items():
            if dataclasses.is_dataclass(key):
                sub_dict = {}
                for key2, value2 in asdict(key):
                    sub_dict[key2] = value2
                main_dict["settings"][key] = sub_dict
            else:
                main_dict["settings"][key] = value
        return main_dict

    def reset_to_default(self):
        self.model_size = ModelSize()
        self.model_size.reset_to_default()
        self.field = Field()
        self.field.reset_to_default()
        self.response = Response()
        self.response.reset_to_default()
        self.localisation = Localisation()
        self.localisation.reset_to_default()
        self.observation = Observation()
        self.observation.reset_to_default()
        self.optional = Optional()
        self.optional.reset_to_default()
        self.case_name = "A"

    def set_observed_cell_indices(self):
        self.observation.selected_grid_cells = cell_index_list_from_obs_positions(
            self.response.grid_dimension,
            self.model_size.size,
            self.observation.obs_positions,
            self.model_size.use_eclipse_grid_index_origo,
        )


def read_config_file(config_file_name: Path) -> Dict[str, Any]:
    with open(config_file_name, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    case_name = settings["settings"]["case_name"]
    model_size = ModelSize(**settings["settings"]["model_size"])
    field = Field(**settings["settings"]["field"])
    response = Response(**settings["settings"]["response"])
    observation = Observation(**settings["settings"]["observation"])
    localisation = Localisation(**settings["settings"]["localisation"])
    optional = Optional(**settings["settings"]["optional"])
    settings = Settings(
        case_name=case_name,
        model_size=model_size,
        field=field,
        response=response,
        observation=observation,
        localisation=localisation,
        optional=optional,
    )
    settings.set_observed_cell_indices()
    return settings


def update_key(key, default_value, settings_dict, parent_key=None):
    value = settings_dict.get(key, default_value)
    if value != default_value:
        if parent_key:
            print(f"Changed settings parameter for {key} under {parent_key} : {value} ")
        else:
            print(f"Changed settings parameter for {key} : {value} ")
    return value


def update_settings(settings_original: Settings, config_dict: dict):
    # pylint: disable=too-many-branches, too-many-statements
    settings_dict = config_dict["settings"]
    settings = copy.deepcopy(settings_original)
    valid_keys = [
        "case_name",
        "model_size",
        "field",
        "response",
        "observation",
        "localisation",
        "optional",
    ]
    for key in settings_dict:
        if key not in valid_keys:
            raise KeyError(f"Unknown keyword {key} in 'settings' ")

    key = "case_name"
    if key in settings_dict:
        settings[key] = settings_dict[key]

    key = "model_size"
    model_size_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = [
        "size",
        "polygon_file",
        "use_eclipse_grid_index_origo",
    ]
    if model_size_dict:
        err_msg = []
        for sub_key in model_size_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        model_size_object = settings.model_size
        model_size_object.size = update_key(
            "size", model_size_object.size, model_size_dict, key
        )
        model_size_object.polygon_file = update_key(
            "polygon_file", model_size_object.polygon_file, model_size_dict, key
        )
        model_size_object.use_eclipse_grid_index_origo = update_key(
            "use_eclipse_grid_index_origo",
            model_size_object.use_eclipse_grid_index_origo,
            model_size_dict,
            key,
        )

    key = "field"
    field_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = [
        "name",
        "algorithm",
        "file_format",
        "initial_file_name_prefix",
        "updated_file_name_prefix",
        "seed_file",
        "variogram",
        "correlation_range",
        "correlation_azimuth",
        "correlation_dip",
        "correlation_exponent",
        "trend_use",
        "trend_params",
        "trend_relstd",
        "grid_dimension",
        "grid_file_name",
    ]
    if field_dict:
        err_msg = []
        for sub_key in field_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        field = settings.field
        field.name = update_key("name", field.name, field_dict, key)
        field.algorithm = update_key("algorithm", field.algorithm, field_dict, key)
        field.file_format = update_key(
            "file_format", field.file_format, field_dict, key
        )
        field.initial_file_name_prefix = update_key(
            "initial_file_name_prefix", field.initial_file_name_prefix, field_dict, key
        )
        field.updated_file_name_prefix = update_key(
            "updated_file_name_prefix", field.updated_file_name_prefix, field_dict, key
        )
        field.seed_file = update_key("seed_file", field.seed_file, field_dict, key)
        field.variogram = update_key("variogram", field.variogram, field_dict, key)
        field.correlation_range = update_key(
            "correlation_range", field.correlation_range, field_dict, key
        )
        field.correlation_azimuth = update_key(
            "correlation_azimuth", field.correlation_azimuth, field_dict, key
        )
        field.correlation_dip = update_key(
            "correlation_dip", field.correlation_dip, field_dict, key
        )
        field.correlation_exponent = update_key(
            "correlation_exponent", field.correlation_exponent, field_dict, key
        )
        field.trend_use = update_key("trend_use", field.trend_use, field_dict, key)
        field.trend_params = update_key(
            "trend_params", field.trend_params, field_dict, key
        )
        field.grid_dimension = update_key(
            "grid_dimension", field.grid_dimension, field_dict, key
        )
        field.grid_file_name = update_key(
            "grid_file_name", field.grid_file_name, field_dict, key
        )

    key = "response"
    response_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = [
        "name",
        "grid_dimension",
        "upscaled_file_name",
        "grid_file_name",
        "file_format",
        "write_upscaled_field",
        "response_function",
        "gen_data_file_prefix",
        "ert_config_template_file",
        "ert_config_file",
    ]
    if response_dict:
        err_msg = []
        for sub_key in response_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        response = settings.response
        response.name = update_key("name", response.name, response_dict, key)
        response.grid_dimension = update_key(
            "grid_dimension", response.grid_dimension, response_dict, key
        )
        response.upscaled_file_name = update_key(
            "upscaled_file_name", response.upscaled_file_name, response_dict, key
        )
        response.grid_file_name = update_key(
            "grid_file_name", response.grid_file_name, response_dict, key
        )
        response.file_format = update_key(
            "file_format", response.file_format, response_dict, key
        )
        response.write_upscaled_field = update_key(
            "write_upscaled_field", response.write_upscaled_field, response_dict, key
        )
        response.response_function = update_key(
            "response_function", response.response_function, response_dict, key
        )
        response.gen_data_file_prefix = update_key(
            "gen_data_file_prefix", response.gen_data_file_prefix, response_dict, key
        )
        response.ert_config_template_file = update_key(
            "ert_config_template_file",
            response.ert_config_template_file,
            response_dict,
            key,
        )
        response.ert_config_file = update_key(
            "ert_config_file",
            response.ert_config_file,
            response_dict,
            key,
        )

    key = "observation"
    obs_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = [
        "directory",
        "file_name",
        "data_dir",
        "reference_param_file",
        "reference_field_name",
        "rel_error",
        "min_abs_error",
        "obs_positions",
    ]

    if obs_dict:
        err_msg = []
        for sub_key in obs_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        observation = settings.observation
        observation.directory = update_key(
            "directory", observation.directory, obs_dict, key
        )
        observation.file_name = update_key(
            "file_name", observation.file_name, obs_dict, key
        )
        observation.data_dir = update_key(
            "data_dir", observation.data_dir, obs_dict, key
        )
        observation.reference_param_file = update_key(
            "reference_param_file", observation.reference_param_file, obs_dict, key
        )
        observation.reference_field_name = update_key(
            "reference_field_name", observation.reference_field_name, obs_dict, key
        )
        observation.rel_error = update_key(
            "rel_error", observation.rel_error, obs_dict, key
        )
        observation.min_abs_error = update_key(
            "min_abs_error", observation.min_abs_error, obs_dict, key
        )
        observation.obs_positions = update_key(
            "obs_positions", observation.obs_positions, obs_dict, key
        )

    key = "localisation"
    local_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = ["method", "use_localisation"]

    if local_dict:
        err_msg = []
        for sub_key in local_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        localisation = settings.localisation
        localisation.method = update_key("method", localisation.method, local_dict, key)
        localisation.use_localisation = update_key(
            "use_localisation", localisation.use_localisation, local_dict, key
        )

    key = "optional"
    optional_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = ["write_obs_pred_diff_field_file"]

    if optional_dict:
        err_msg = []
        for sub_key in optional_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        optional = settings.optional
        optional.write_obs_pred_diff_field_file = update_key(
            "write_obs_pred_diff_field_file",
            optional.write_obs_pred_diff_field_file,
            optional_dict,
            key,
        )
    settings.set_observed_cell_indices()
    return settings


def generate_field_and_upscale(
    # pylint: disable=too-many-arguments
    real_number: int,
    iteration: int,
    seed_file_name: str,
    algorithm_method: str,
    field_name: str,
    field_file_name: str,
    file_format: str,
    grid_dimension: tuple,
    model_size: tuple,
    variogram_name: str,
    corr_ranges: tuple,
    azimuth: float,
    dip: float,
    alpha: float,
    use_trend: bool,
    trend_params: tuple,
    relative_std: float,
    upscale_name: str,
    response_function: str,
    upscaled_file_name: str,
    grid_dimension_upscaled: tuple,
    write_upscaled_field: bool,
    use_standard_grid_index_origo: bool,
):
    start_seed = get_seed(seed_file_name, real_number)
    if algorithm_method == "gstools":
        print(f"Use algorithm: {algorithm_method}  with version: {gs.__version__} ")
        residual_field = simulate_field_using_gstools(
            start_seed,
            variogram_name,
            corr_ranges,
            azimuth,
            grid_dimension,
            model_size,
            use_standard_grid_index_origo,
        )
    else:
        print("Use algorithm: gaussianfft")
        residual_field = simulate_field(
            start_seed,
            variogram_name,
            corr_ranges,
            azimuth,
            dip,
            alpha,
            grid_dimension,
            model_size,
            use_standard_grid_index_origo,
        )
    if use_trend:
        trend_field = trend(grid_dimension, model_size, trend_params)
        field3D = trend_field + relative_std * residual_field
    else:
        field3D = residual_field

    # Write field parameter for fine scale grid
    field_object = export_field(
        field3D, field_name, field_file_name, file_format, grid_dimension
    )

    field_values = field_object.values

    # Calculate upscaled values for selected coarse grid cells
    upscaled_values = upscaling(
        field_values,
        response_function,
        file_format,
        upscale_name,
        write_upscaled_field,
        upscaled_file_name,
        grid_dimension_upscaled,
        iteration,
    )
    return upscaled_values


def get_seed(seed_file_name, r_number):
    with open(seed_file_name, "r", encoding="utf8") as file:
        lines = file.readlines()
        try:
            seed_value = int(lines[r_number])
        except IndexError as exc:
            raise IOError("Seed value not found for realization {r_number}  ") from exc
        except ValueError as exc:
            raise IOError(
                "Invalid seed value in file for realization{r_number}"
            ) from exc
    return seed_value


def upscaling(
    # pylint: disable=too-many-arguments
    field_values,
    response_function_name: str,
    file_format: str,
    upscaled_field_name: str,
    write_upscaled_field: bool,
    upscaled_file_name: str,
    dimension: tuple,
    iteration: int = 0,
):
    """
    Calculate upscaled values and optionally write upscaled values to file.
    Return upscaled values
    """
    NX, NY, NZ = dimension
    upscaled_values = np.zeros((NX, NY, NZ), dtype=np.float32, order="F")
    upscaled_values[:, :, :] = -999

    if response_function_name == "average":
        upscaled_values = upscale_average(
            field_values,
            upscaled_values,
        )

    if write_upscaled_field:
        if iteration == 0:
            upscaled_file_name = "init_files/" + upscaled_file_name

        write_upscaled_field_to_file(
            upscaled_values,
            upscaled_file_name,
            upscaled_field_name,
            file_format=file_format,
        )

    return upscaled_values


# pylint: disable=too-many-arguments
def write_upscaled_field_to_file(
    upscaled_values,
    upscaled_file_name,
    upscaled_field_name,
    selected_cell_index_list=None,
    file_format="ROFF",
):
    nx, ny, nz = upscaled_values.shape

    field_object = xtgeo.grid3d.GridProperty(
        ncol=nx,
        nrow=ny,
        nlay=nz,
        values=upscaled_values,
        discrete=False,
        name=upscaled_field_name,
    )

    if file_format.upper() == "ROFF":
        fullfilename = upscaled_file_name + ".roff"
        field_object.to_file(fullfilename, fformat="roff")
    elif file_format.upper() == "GRDECL":
        fullfilename = upscaled_file_name + ".GRDECL"
        field_object.to_file(fullfilename, fformat="grdecl")
    else:
        raise ValueError(f"Unknown file format: {file_format} ")
    print(f"Write upscaled field file: {fullfilename}  ")

    if selected_cell_index_list is not None:
        # Grid index order to xtgeo must be c-order masked array
        selected_upscaled_values = np.ma.zeros((nx, ny, nz), dtype=np.float32)
        selected_upscaled_values[:, :, :] = -1
        nobs = get_nobs_from_cell_index_list(selected_cell_index_list)
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(
                obs_number, selected_cell_index_list
            )
            selected_upscaled_values[Iindx, Jindx, Kindx] = upscaled_values[
                Iindx, Jindx, Kindx
            ]

        field_name_selected = upscaled_field_name + "_conditioned_cells"
        file_name_selected = "init_files/" + field_name_selected + ".roff"
        cond_field_object = xtgeo.grid3d.GridProperty(
            ncol=nx,
            nrow=ny,
            nlay=nz,
            values=selected_upscaled_values,
            discrete=False,
            name=field_name_selected,
        )
        print(f"Write conditioned cell values as field: {file_name_selected}")
        cond_field_object.to_file(file_name_selected, fformat="roff")

    return field_object


def upscale_average(field_values, upscaled_values):
    """
    Input: field_values (numpy 3D)
              coarse_cell_index_list (list of tuples (I,J,K))
    Output: upscaled_values  (numpy 3D) initialized outside
              but filled in specified (I,J,K) cells.
    """
    nx, ny, nz = field_values.shape
    NX, NY, NZ = upscaled_values.shape

    print(f"Number of fine scale grid cells:   (nx,ny,nz): ({nx},{ny},{nz})")
    print(f"Number of coarse scale grid cells: (NX,NY,NZ): ({NX},{NY},{NZ})  ")
    mx = int(nx / NX)
    my = int(ny / NY)
    mz = int(nz / NZ)
    print(
        "Number of fine scale grid cells per coarse grid cell: "
        f"(mx,my,mz): ({mx},{my},{mz})    "
    )

    print("Calculate upscaled values for all grid cells")
    for Kindx in range(NZ):
        for Jindx in range(NY):
            for Iindx in range(NX):
                i_slice = slice(Iindx * mx, (Iindx + 1) * mx)
                j_slice = slice(Jindx * my, (Jindx + 1) * my)
                k_slice = slice(Kindx * mz, (Kindx + 1) * mz)
                upscaled_values[Iindx, Jindx, Kindx] = np.mean(
                    field_values[i_slice, j_slice, k_slice]
                )
    return upscaled_values


def trend(grid_dimension: tuple, model_size: tuple, trend_params: tuple):
    """
    Return 3D numpy array with values following a linear trend
    scaled to take values between 0 and 1.
    """
    nx, ny, nz = grid_dimension
    xsize, ysize, _ = model_size
    a, b = trend_params

    x0 = 0.0
    y0 = 0.0
    dx = xsize / nx
    dy = ysize / ny

    maxsize = ysize
    if xsize > ysize:
        maxsize = xsize

    val = np.zeros((nx, ny, nz), dtype=np.float32, order="F")
    for i in range(nx):
        x = x0 + i * dx
        for j in range(ny):
            y = y0 + j * dy
            for k in range(nz):
                val[i, j, k] = a * (x - x0) / maxsize + b * (y - y0) / maxsize

    minval = np.min(val)
    maxval = np.max(val)
    val_normalized = (val - minval) / (maxval - minval)
    return val_normalized


def simulate_field(
    start_seed: int,
    variogram_name: str,
    corr_ranges: tuple,
    azimuth: float,
    dip: float,
    alpha: float,
    grid_dimension: tuple,
    model_size: tuple,
    use_standard_grid_index_origo: bool,
):
    # pylint: disable=import-outside-toplevel
    # This function will not be available untill gaussianfft is available on python 3.10
    # import gaussianfft as sim  # isort: skip
    # dummy code to avoid pylint complaining in github actions
    sim = None
    nx, ny, nz = grid_dimension
    xrange, yrange, zrange = corr_ranges
    xsize, ysize, zsize = model_size

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    print(f"Start seed: {start_seed}")
    sim.seed(start_seed)

    variogram = sim.variogram(
        variogram_name,
        xrange,
        perp_range=yrange,
        depth_range=zrange,
        azimuth=azimuth - 90,
        dip=dip,
        power=alpha,
    )

    print(f"Simulate field with size: nx={nx},ny={ny} ")
    # gaussianfft.simulate  will save the values in F-order
    field1D = sim.simulate(variogram, nx, dx, ny, dy, nz, dz)
    field_sim = field1D.reshape((nx, ny, nz), order="F")
    if use_standard_grid_index_origo:
        field_c_order = np.ma.zeros((nx, ny, nz), dtype=np.float32)
        j_indices = -np.arange(ny) + ny - 1
        # Flip j index and use c-order
        field_c_order[:, j_indices, :] = field_sim[:, :, :]
        return field_c_order
    # Change to C-order
    field_c_order = np.ma.zeros((nx, ny, nz), dtype=np.float32)
    field_c_order[:, :, :] = field_sim[:, :, :]
    return field_c_order


def simulate_field_using_gstools(
    start_seed: int,
    variogram_name: str,
    corr_ranges: tuple,
    azimuth: float,
    grid_dimension: tuple,
    model_size: tuple,
    use_standard_grid_index_origo: bool,
):
    # pylint: disable=no-member,
    xrange, yrange, zrange = corr_ranges
    xsize, ysize, zsize = model_size
    nx, ny, nz = grid_dimension

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    x = np.arange(0.5 * dx, xsize, dx)
    y = np.arange(0.5 * dy, ysize, dy)
    z = np.arange(0.5 * dz, zsize, dz)
    # Rescale factor is set to:
    #    sqrt(3.0) for gaussian correlation functions,
    #    3.0 for exponetial correlation function,
    # to ensure the correlation function have the same definition of correlation
    # lenght as is used in RMS and gaussianfft algorithm.
    print(f"Variogram name:  {variogram_name}")
    if variogram_name.upper() == "GAUSSIAN":
        model = gs.Gaussian(
            dim=3,
            var=1.0,
            len_scale=[xrange, yrange, zrange],
            angles=np.pi * (0.5 - azimuth / 180.0),
            rescale=math.sqrt(3),
        )
    elif variogram_name.upper() == "EXPONENTIAL":
        model = gs.Exponential(
            dim=3,
            var=1.0,
            len_scale=[xrange, yrange, zrange],
            angles=np.pi * (0.5 - azimuth / 180.0),
            rescale=3,
        )
    else:
        raise ValueError(f"Unknown variogram type: {variogram_name} ")

    print(f"Start seed: {start_seed}")
    print(f"Simulate field with size: nx={nx},ny={ny} nz={nz} ")
    srf = gs.SRF(model, seed=start_seed)
    field_srf = srf.structured([x, y, z], store="Field")
    field = field_srf.reshape((nx, ny, nz))
    if use_standard_grid_index_origo:
        field_flip_j_index = np.ma.zeros((nx, ny, nz), dtype=np.float32)
        j_indices = -np.arange(ny) + ny - 1
        field_flip_j_index[:, j_indices, :] = field[:, :, :]
        return field_flip_j_index

    return field


def export_field(
    field3D,
    field_name: str,
    field_file_name: str,
    file_format: str,
    grid_dimension: tuple,
):
    """
    Export initial realization of field to roff format
    Input field3D should be C-index order since xtgeo requires that
    """
    nx, ny, nz = grid_dimension
    field_object = xtgeo.grid3d.GridProperty(
        ncol=nx, nrow=ny, nlay=nz, values=field3D, discrete=False, name=field_name
    )
    if file_format.upper() == "GRDECL":
        fullfilename = field_file_name + ".grdecl"
        field_object.to_file(
            fullfilename, fformat="grdecl", dtype="float32", fmt="%20.6f"
        )
    elif file_format.upper() == "ROFF":
        fullfilename = field_file_name + ".roff"
        field_object.to_file(fullfilename, fformat="roff")
    else:
        raise IOError(f"Unknown file format for fields: {file_format} ")
    print(f"Write field file: {fullfilename}  ")
    return field_object


def read_field_from_file(
    input_file_name: str, name: str, file_format: str, grid_file_name: str
):
    """
    Read field from roff formatted file and return xtgeo property object
    """
    if file_format.upper() == "GRDECL":
        grid = xtgeo.grid_from_file(grid_file_name, fformat="egrid")
        fullfilename = input_file_name + ".grdecl"
        field_object = xtgeo.grid3d.GridProperty(
            fullfilename, fformat="grdecl", grid=grid, name=name
        )
    elif file_format.upper() == "ROFF":
        fullfilename = input_file_name + ".roff"
        field_object = xtgeo.gridproperty_from_file(
            fullfilename, fformat="roff", name=name
        )
    else:
        raise IOError(f"Unknown file format for fields: {file_format} ")
    return field_object


def read_obs_field_from_file(
    file_format: str, pred_obs_file_name: str, grid_file_name: str, field_name: str
):
    """
    Read field parameter containing parameter with observed values
    for selected grid cells
    """
    if file_format.upper() == "ROFF":
        fullfilename = pred_obs_file_name + ".roff"
        obs_field_object = xtgeo.gridproperty_from_file(fullfilename, fformat="roff")
    elif file_format.upper() == "GRDECL":
        grid = xtgeo.grid_from_file(grid_file_name, fformat="egrid")
        fullfilename = pred_obs_file_name + ".grdecl"
        obs_field_object = xtgeo.gridproperty_from_file(
            fullfilename, fformat="grdecl", grid=grid, name=field_name
        )
    else:
        raise IOError(f"Unknown file format:{file_format} ")
    return obs_field_object


def read_upscaled_field_from_file(
    iteration: int,
    input_file_name: str,
    file_format: str,
    upscaled_field_name: str,
    grid_file_name: str,
):
    """
    Read upscaled field parameter either from initial ensemble or updated ensemble.
    Return xtgeo property object
    """

    if iteration == 0:
        filename = "init_files/" + input_file_name
    else:
        filename = input_file_name
    if file_format.upper() == "ROFF":
        fullfilename = filename + ".roff"
        field_object = xtgeo.gridproperty_from_file(fullfilename, fformat="roff")
    elif file_format.upper() == "GRDECL":
        grid = xtgeo.grid_from_file(grid_file_name, fformat="egrid")
        fullfilename = filename + ".grdecl"
        field_object = xtgeo.gridproperty_from_file(
            fullfilename, fformat="grdecl", grid=grid, name=upscaled_field_name
        )
    else:
        raise IOError(f"Unknown file format:{file_format} ")
    return field_object


def write_obs_pred_diff_field(
    upscaled_field_object, observation_field_object, file_format: str
):
    """
    Get xtgeo property objects for predicted values for observables
    and observation values.
    Write file with difference as roff formatted file.
    """
    nx, ny, nz = upscaled_field_object.dimensions
    values_diff = upscaled_field_object.values - observation_field_object.values

    diff_object = xtgeo.grid3d.GridProperty(
        ncol=nx,
        nrow=ny,
        nlay=nz,
        values=values_diff,
        discrete=False,
        name="DiffObsPred",
    )

    filename = "DiffObsPred"
    if file_format.upper() == "ROFF":
        fullfilename = filename + ".roff"
        diff_object.to_file(fullfilename, fformat="roff")
    elif file_format.upper() == "GRDECL":
        fullfilename = filename + ".grdecl"
        diff_object.to_file(fullfilename, fformat="grdecl")
    else:
        raise IOError(f"Unknown file format: {file_format} ")
    print(
        "Write field with difference between upscaled reference "
        "field from which observations are extracted and "
        f"and prediction from current realization: {fullfilename}  "
    )


def get_cell_indices(obs_number, cell_indx_list):
    if cell_indx_list is None:
        return None
    if len(cell_indx_list) == 0:
        return None
    try:
        Iindx = cell_indx_list[obs_number][0]
        Jindx = cell_indx_list[obs_number][1]
        Kindx = cell_indx_list[obs_number][2]
    except IndexError:
        try:
            Iindx = cell_indx_list[0]
            Jindx = cell_indx_list[1]
            Kindx = cell_indx_list[2]
        except IndexError:
            raise ValueError(f"Index error for cell indices in:  {cell_indx_list}")

    return (Iindx, Jindx, Kindx)


def get_obs_pos(obs_number, obs_pos_list):
    x = obs_pos_list[obs_number][0]
    y = obs_pos_list[obs_number][1]
    z = obs_pos_list[obs_number][2]

    return (x, y, z)


def get_nobs_from_cell_index_list(cell_index_list: list):
    """
    Check if cell_index_list is a single tuple (i,j,k) or
    a list of tuples  of type (i,j,k).
    Return number of cell_indices found in list
    """
    is_list_of_ints = all(isinstance(indx, int) for indx in cell_index_list)
    if is_list_of_ints:
        nobs = 1
    else:
        # list of tuples (x,y,z)
        nobs = len(cell_index_list)
    return nobs


def get_nobs_from_position_list(obs_positions: list):
    """
    Check if obs_positions is a single tuple (x,y,z) or
    a list of tuples  of type (x,y,z).
    Return number of positions found in list
    """
    if not obs_positions:
        return 0
    is_list_of_floats = all(
        isinstance(coordinate, float) for coordinate in obs_positions
    )
    if is_list_of_floats:
        nobs = 1
    else:
        # list of tuples (x,y,z)
        nobs = len(obs_positions)
    return nobs


def obs_positions_from_cell_index(
    grid_dimension_upscaled: tuple,
    model_size: tuple,
    cell_indx_list: list,
    use_eclipse_origo: bool,
):
    nx, ny, nz = grid_dimension_upscaled
    xsize, ysize, zsize = model_size

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    if use_eclipse_origo:
        print("Grid index origin: Eclipse standard")
    else:
        print("Grid index origin: RMS standard")
    print(
        "Observation reference point coordinates is always "
        "from origin at lower left corner"
    )

    pos_list = []
    nobs = get_nobs_from_cell_index_list(cell_indx_list)
    for obs_number in range(nobs):
        (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, cell_indx_list)
        x = (Iindx + 0.5) * dx
        z = (Kindx + 0.5) * dz
        if use_eclipse_origo:
            y = ysize - (Jindx + 0.5) * dy
        else:
            y = (Jindx + 0.5) * dy

        pos_list.append((x, y, z))

    return pos_list


def cell_index_list_from_obs_positions(
    grid_dimension_upscaled: tuple,
    model_size: tuple,
    obs_positions: list,
    use_eclipse_origo: bool,
):
    nx, ny, nz = grid_dimension_upscaled
    xsize, ysize, zsize = model_size

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    if use_eclipse_origo:
        print("Grid index origin: Eclipse standard")
    else:
        print("Grid index origin: RMS standard")
    print(
        "Observation reference point coordinates is always "
        "from origin at lower left corner"
    )

    cell_index_list = []
    nobs = get_nobs_from_position_list(obs_positions)
    for obs_number in range(nobs):
        obs_pos = get_obs_pos(obs_number, obs_positions)
        Iindx = int(obs_pos[0] / dx)
        Jindx = int(obs_pos[1] / dy)
        Kindx = int(obs_pos[2] / dz)
        if use_eclipse_origo:
            Jindx = ny - Jindx - 1
        cell_index_list.append((Iindx, Jindx, Kindx))

    return cell_index_list


def read_observations(
    config_path: str,
    observation_dir: str,
    obs_data_dir: str,
    cell_index_list: list,
):
    values = []
    for index in cell_index_list:
        Iindx = index[0]
        Jindx = index[1]
        Kindx = index[2]
        filename = (
            config_path
            + "/"
            + observation_dir
            + "/"
            + obs_data_dir
            + "/"
            + "obs_"
            + str(Iindx + 1)
            + "_"
            + str(Jindx + 1)
            + "_"
            + str(Kindx + 1)
            + ".txt"
        )
        print(f"Read observations from: {filename}")
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            words = line.split()
            if len(words) == 2:
                values.append(float(words[0]))
    return values


def write_gen_obs(
    upscaled_values,
    observation_dir: str,
    obs_file_name: str,
    obs_data_dir: str,
    cell_index_list: list,
    rel_err: float,
    min_err: float,
):
    if os.path.exists(observation_dir):
        shutil.rmtree(observation_dir)
    print(f"Create directory: {observation_dir} ")
    os.makedirs(observation_dir)
    data_dir = observation_dir + "/" + obs_data_dir
    if not os.path.exists(data_dir):
        print(f"Create directory: {data_dir} ")
        os.makedirs(data_dir)

    print(f"Write observation file: {obs_file_name} ")
    filename = observation_dir + "/" + obs_file_name
    observation_value_list = []
    with open(filename, "w", encoding="utf8") as obs_file:
        # Check if obs_position list is a single tuple (i,j,k)
        # or a list of tuples  of type (i,j,k)
        nobs = get_nobs_from_cell_index_list(cell_index_list)
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, cell_index_list)
            value = upscaled_values[Iindx, Jindx, Kindx]
            observation_value_list.append(value)
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


def create_grid(
    grid_file_name,
    dimensions,
    size,
    standard_grid_index_origo,
    polygon_file_name=None,
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


# pylint: disable=too-many-statements,too-many-branches
def write_localisation_config(
    obs_positions: list,
    obs_index_list: list,
    field_name: str,
    corr_ranges: tuple,
    azimuth: float,
    config_file_name: str = "local_config.yml",
    write_scaling: bool = True,
    localisation_method: str = "gaussian",
    segment_file_name: str = None,
    scaling_param_file_name: str = None,
):
    space = " " * 2
    space2 = " " * 4
    space3 = " " * 6
    print(f"Write localisation config file: {config_file_name}")
    with open(config_file_name, "w", encoding="utf8") as file:
        file.write("log_level: 3\n")
        file.write(f"write_scaling_factors: {write_scaling}\n")
        file.write("correlations:\n")
        nobs = get_nobs_from_position_list(obs_positions)
        if nobs != get_nobs_from_cell_index_list(obs_index_list):
            raise ValueError(
                "Inconsistency between observation list and cell-index_list"
            )
        local_method = localisation_method.lower()
        if local_method in ["gaussian", "exponential"]:
            for obs_number in range(nobs):
                (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, obs_index_list)
                obs_name = f"OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}"
                pos = obs_positions[obs_number]
                file.write(f"{space}- name: CORR_{obs_number}\n")
                file.write(f"{space2}obs_group:\n")
                file.write(f'{space3}add: ["{obs_name}"]\n')
                file.write(f"{space2}param_group:\n")
                file.write(f'{space3}add: ["{field_name}"]\n')
                file.write(f"{space2}field_scale:\n")
                if local_method == "gaussian":
                    file.write(f"{space3}method: gaussian_decay\n")
                else:
                    file.write(f"{space3}method: exponential_decay\n")
                file.write(f"{space3}main_range: {corr_ranges[0]}\n")
                file.write(f"{space3}perp_range: {corr_ranges[1]}\n")
                file.write(f"{space3}azimuth: {azimuth}\n")
                file.write(f"{space3}ref_point: [ {pos[0]}, {pos[1]} ]\n")

        if local_method == "constant":
            # Constant scaling factor, use only one correlation group
            file.write(f"{space}- name: CORR\n")
            file.write(f"{space2}obs_group:\n")
            file.write(f"{space3}add: [ ")
            obs_list = ""
            for obs_number in range(nobs):
                (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, obs_index_list)
                obs_name = f' "OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}" '
                obs_list += obs_name
                if obs_number < (nobs - 1):
                    obs_list += ", "
                else:
                    obs_list += " ]"
            file.write(obs_list)
            file.write("\n")
            pos = obs_positions[0]
            file.write(f"{space2}param_group:\n")
            file.write(f'{space3}add: ["{field_name}"]\n')
            file.write(f"{space2}field_scale:\n")
            file.write(f"{space3}method: constant\n")
            file.write(f"{space3}value: 1.0\n")

        if local_method == "region":
            # Use region parameter with one obs per region
            if segment_file_name is None or len(segment_file_name) == 0:
                raise ValueError("Missing segment file name when using region")

            for obs_number in range(nobs):
                (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, obs_index_list)
                obs_name = f"OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}"
                pos = obs_positions[obs_number]
                file.write(f"{space}- name: CORR_{obs_number}\n")
                file.write(f"{space2}obs_group:\n")
                file.write(f'{space3}add: ["{obs_name}"]\n')
                file.write(f"{space2}param_group:\n")
                file.write(f'{space3}add: ["{field_name}"]\n')
                file.write(f"{space2}field_scale:\n")
                file.write(f"{space3}method: segment\n")
                file.write(f'{space3}segment_filename: "{segment_file_name}"\n')
                file.write(f'{space3}param_name: "Region"\n')
                file.write(f"{space3}active_segments: [ {obs_number + 1} ]\n")
                file.write(f"{space3}scalingfactors: [ 1.0 ]\n")
                file.write(f"{space3}smooth_ranges: [ 1, 1 ]\n")

        if local_method == "scaling_file":
            # Use a scaling factor from file
            file.write(f"{space}- name: CORR\n")
            file.write(f"{space2}obs_group:\n")
            file.write(f"{space3}add: [ ")
            obs_list = ""
            for obs_number in range(nobs):
                (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, obs_index_list)
                obs_name = f' "OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}" '
                obs_list += obs_name
                if obs_number < (nobs - 1):
                    obs_list += ", "
                else:
                    obs_list += " ]"
            file.write(obs_list)
            file.write("\n")
            pos = obs_positions[0]
            file.write(f"{space2}param_group:\n")
            file.write(f'{space3}add: ["{field_name}"]\n')
            file.write(f"{space2}field_scale:\n")
            file.write(f"{space3}method: from_file\n")
            file.write(f'{space3}filename:  "{scaling_param_file_name}"\n')
            file.write(f'{space3}param_name: "SCALING"\n')


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


def generate_segments(
    polygon_file_name: str,
    region_file_name: str,
    grid_object,
):
    if polygon_file_name is None or not os.path.exists(polygon_file_name):
        raise ValueError(
            f"Missing polygon file  {polygon_file_name} when creating region parameter"
        )
    dimensions = grid_object.dimensions
    nx = dimensions[0]
    ny = dimensions[1]
    nz = dimensions[2]
    codenames_dict = {
        1: "RegionA",
        2: "RegionB",
        3: "RegionC",
    }
    region_object = xtgeo.GridProperty(
        ncol=nx,
        nrow=ny,
        nlay=nz,
        name="Region",
        discrete=True,
        values=np.zeros((nx, ny, nz), dtype=np.int32),
        codes=codenames_dict,
        grid=grid_object,
    )

    print(f"Use polygon file {polygon_file_name} to create region parameter ")
    polygon_object = xtgeo.polygons_from_file(polygon_file_name, fformat="xyz")
    for poly_id in [0, 1, 2]:
        wpoly = polygon_object.copy()
        wpoly.dataframe = wpoly.dataframe[wpoly.dataframe.POLY_ID == poly_id]
        region_object.set_inside(wpoly, poly_id + 1)

    print(f"Write region parameter: {region_file_name} ")
    region_object.to_file(region_file_name, fformat="grdecl")


def create_ert_config_file(
    template_file_name: str,
    case_name: str,
    ensemble_seed_file_name: str,
    nreal: int,
    ert_start_seed: int,
    response_file_name_prefix: str,
    grid_file_name: str,
    upscaled_grid_file_name: str,
    initial_field_param_file_prefix: str,
    updated_field_param_file_prefix: str,
    field_file_format: str,
    use_localisation: bool,
    output_config_file_name: str = "sim_field.ert",
) -> None:
    try:
        with open(template_file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except IOError:
        raise IOError(f"Cannot open and read file:  {template_file_name}")

    load_workflow = "LOAD_WORKFLOW localisation.wf  LOCALISATION_WORKFLOW"
    hook_workflow = "HOOK_WORKFLOW  LOCALISATION_WORKFLOW    PRE_FIRST_UPDATE"
    local = "_local" if use_localisation else ""

    strings_to_replace_dict = {
        "CASE_NAME": "sim_field_" + case_name + local,
        "ENSEMBLE_SEED_FILE": ensemble_seed_file_name,
        "NREAL": str(nreal),
        "ERT_START_SEED": str(ert_start_seed),
        "RESPONSE_FILE_PREFIX": response_file_name_prefix,
        "GRID_FILE_NAME": grid_file_name,
        "GRID_FILE_NAME_UPSCALED": upscaled_grid_file_name,
        "INITIAL_FIELDPARAM_FILE_NAME": initial_field_param_file_prefix
        + "."
        + field_file_format.lower(),
        "UPDATED_FIELDPARAM_FILE_NAME": updated_field_param_file_prefix
        + "."
        + field_file_format.lower(),
        "LOAD_LOCALISATION_WORKFLOW": load_workflow
        if use_localisation
        else "-- " + load_workflow,
        "HOOK_LOCALISATION_WORKFLOW": hook_workflow
        if use_localisation
        else "-- " + hook_workflow,
    }
    template_file_contents = Template("".join(lines))
    updated_contents = template_file_contents.safe_substitute(strings_to_replace_dict)

    if use_localisation:
        print(f"Write ERT config file with localisation: {output_config_file_name}")
    else:
        print(f"Write ERT config file: {output_config_file_name}")
    with open(output_config_file_name, "w", encoding="utf-8") as file:
        file.write(updated_contents)


def initialize_case(settings):
    # Create seed file
    generate_seed_file(settings.field.seed_file)

    # Create grid for the field parameter
    grid_object = create_grid(
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

    # Simulate field (with trend)
    real_number = 0
    iteration = 0
    upscaled_values = generate_field_and_upscale(
        real_number,
        iteration,
        settings.field.seed_file,
        settings.field.algorithm,
        settings.field.name,
        settings.field.initial_file_name_prefix,
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
    print(
        "Selected grid cells having observations: "
        f"{settings.observation.selected_grid_cells}"
    )
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
        settings.observation.obs_positions,
        settings.observation.selected_grid_cells,
        settings.field.name,
        settings.field.correlation_range,
        settings.field.correlation_azimuth,
        config_file_name="local_config.yml",
        write_scaling=True,
        localisation_method=settings.localisation.method,
        segment_file_name=settings.localisation.region_file,
        scaling_param_file_name=settings.localisation.scaling_file,
    )

    if settings.localisation.method == "region":
        generate_segments(
            settings.localisation.region_polygons,
            settings.localisation.region_file,
            grid_object,
        )

    create_ert_config_file(
        settings.response.ert_config_template_file,
        settings.case_name,
        settings.field.seed_file,
        10,
        59716487,
        settings.response.gen_data_file_prefix,
        settings.field.grid_file_name,
        settings.response.grid_file_name,
        settings.field.initial_file_name_prefix,
        settings.field.updated_file_name_prefix,
        settings.field.file_format,
        settings.localisation.use_localisation,
        settings.response.ert_config_file,
    )


def example_cases(name):
    """
    Define the different cases here.
    They define how to modify default settings
    for the varios cases to test.
    """
    if name == "A":
        params = {
            "case_name": "A",
            "model_size": {
                "use_eclipse_grid_index_origo": True,
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD_UPSCALED.EGRID",
            },
            "observation": {
                "obs_positions": [
                    [750.0, 750.0, 25.0],
                    [250.0, 1750.0, 25.0],
                    [250.0, 250.0, 25.0],
                ],
            },
            "localisation": {
                "use_localisation": True,
            },
        }
    elif name == "B":
        params = {
            "case_name": "B",
            "model_size": {
                "use_eclipse_grid_index_origo": True,
            },
            "field": {
                "file_format": "GRDECL",
                "grid_file_name": "GRID_STANDARD.EGRID",
            },
            "response": {
                "file_format": "GRDECL",
                "grid_file_name": "GRID_STANDARD_UPSCALED.EGRID",
            },
            "observation": {
                "obs_positions": [[750.0, 750.0, 25.0], [250.0, 1750.0, 25.0]],
            },
            "localisation": {
                "use_localisation": True,
            },
        }
    elif name == "C":
        params = {
            "case_name": "C",
            "model_size": {
                "use_eclipse_grid_index_origo": False,
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_RMS_ORIGO.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_RMS_ORIGO_UPSCALED.EGRID",
            },
            "observation": {
                "obs_positions": [[750.0, 750.0, 25.0], [250.0, 1750.0, 25.0]],
            },
            "localisation": {
                "use_localisation": True,
            },
        }
    elif name == "A2":
        params = {
            "case_name": "A2",
            "model_size": {
                "use_eclipse_grid_index_origo": True,
                "polygon_file": "init_files/polygons.txt",
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_WITH_ACTNUM.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "UpscaleGrid.EGRID",
            },
            "observation": {
                "obs_positions": [[750.0, 750.0, 25.0], [250.0, 1750.0, 25.0]],
            },
            "localisation": {
                "use_localisation": True,
            },
        }
    elif name == "D":
        params = {
            "case_name": "D",
            "model_size": {
                "use_eclipse_grid_index_origo": True,
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD_UPSCALED.EGRID",
            },
            "localisation": {
                "method": "constant",
                "use_localisation": True,
            },
            "observation": {
                "obs_positions": [[750.0, 750.0, 25.0], [250.0, 1750.0, 25.0]],
            },
        }
    elif name == "E":
        params = {
            "case_name": "E",
            "model_size": {
                "use_eclipse_grid_index_origo": True,
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD_UPSCALED.EGRID",
            },
            "localisation": {
                "method": "region",
                "region_polygons": "init_files/region_polygons.txt",
                "region_file": "regions.grdecl",
                "use_localisation": True,
            },
            "observation": {
                "obs_positions": [[450.0, 1750.0, 25.0], [250.0, 250.0, 25.0]],
            },
        }
    elif name == "F":
        params = {
            "case_name": "F",
            "model_size": {
                "use_eclipse_grid_index_origo": True,
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_STANDARD_UPSCALED.EGRID",
            },
            "localisation": {
                "method": "scaling_file",
                "scaling_file": "init_files/scaling_factor.grdecl",
                "use_localisation": True,
            },
            "observation": {
                "obs_positions": [[650.0, 850.0, 25.0]],
            },
        }
    elif name == "G":
        params = {
            "case_name": "G",
            "model_size": {
                "use_eclipse_grid_index_origo": False,
                "polygon_file": None,
            },
            "field": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_RMS_ORIGO.EGRID",
            },
            "response": {
                "file_format": "ROFF",
                "grid_file_name": "GRID_RMS_ORIGO_UPSCALED.EGRID",
            },
            "localisation": {
                "method": "scaling_file",
                "scaling_file": "init_files/scaling_factor_rms_origo.grdecl",
                "use_localisation": True,
            },
            "observation": {
                "obs_positions": [[650.0, 850.0, 25.0]],
            },
        }
    return params
