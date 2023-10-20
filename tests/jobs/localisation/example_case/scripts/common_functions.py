"""
Common functions used by the scripts: init_test_case.py and sim_field.py
"""
import math
import os
from dataclasses import dataclass
from typing import Tuple

import gstools as gs
import numpy as np
import xtgeo
import yaml

# pylint: disable=missing-function-docstring, too-many-locals, invalid-name
# pylint: disable=raise-missing-from
# pylint: disable=too-many-nested-blocks


# Settings for the test case in the following dataclasses
@dataclass
class GridSize:
    """
    Length, width, thickness of a box containing the field
    Same size is used for both fine scale grid with
    the simulated field and the coarse scale grid
    containing upscaled values of the simulated field.
    """

    xsize: float = 7500.0
    ysize: float = 12500.0
    #    xsize: float = 7500.0
    #    ysize: float = 7500.0
    zsize: float = 50.0
    polygon_file: str = None
    use_eclipse_grid_index_origo: bool = True


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
    initial_file_name: str = "init_files/FieldParam"
    updated_file_name: str = "FieldParam"
    seed_file: str = "randomseeds.txt"
    variogram: str = "exponential"
    correlation_range: Tuple[float] = (3000.0, 1000.0, 2.0)
    correlation_azimuth: float = 45.0
    correlation_dip: float = 0.0
    correlation_exponent: float = 1.9
    trend_use: bool = False
    trend_params: Tuple[float] = (1.0, -1.0)
    trend_relstd: float = 0.05
    grid_dimension: Tuple[int] = (150, 250, 1)
    grid_file_name: str = "GRID.EGRID"


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
    grid_dimension: Tuple[int] = (15, 25, 1)
    upscaled_file_name: str = "Upscaled"
    grid_file_name: str = "UpscaleGrid.EGRID"
    file_format: str = "ROFF"
    write_upscaled_field: bool = True
    response_function: str = "average"
    gen_data_file_name: str = "UpscaledField_0.txt"


# pylint: disable=too-many-instance-attributes
@dataclass
class Observation:
    """
    Specify name of files for generated observations
    and also which grid cells from coarse grid is used
    as observables. (Cells that have values that are used as observations)
    """

    directory: str = "observations"
    file_name: str = "observations.obs"
    data_dir: str = "obs_data"
    reference_param_file: str = "init_files/ObsField"
    reference_field_name: str = "ObsField"
    rel_error: float = 0.10
    min_abs_error: float = 0.01
    # selected_grid_cells: Tuple[Tuple[int]] = ((5, 10, 1), (10, 5, 1))
    selected_grid_cells: Tuple[Tuple[int]] = (8, 12, 1)


@dataclass
class Localisation:
    """
    Specify settings for the localisation config file.
    """

    method: str = "gaussian"
    # method: str = "constant"


@dataclass
class Optional:
    """
    Specify if some optional files should be
    written or not (for QC purpose).
    """

    write_obs_pred_diff_field_file: bool = True


@dataclass
class Settings:
    """
    Settings for the test case
    """

    grid_size: GridSize = GridSize()
    field: Field = Field()
    response: Response = Response()
    observation: Observation = Observation()
    localisation: Localisation = Localisation()
    optional: Optional = Optional()


settings = Settings()


def read_config_file(config_file_name=None):
    # Modify default settings  if config_file exists
    if config_file_name:
        if os.path.exists(config_file_name):
            with open(config_file_name, "r", encoding="utf-8") as yml_file:
                settings_yml = yaml.safe_load(yml_file)

            update_settings(settings_yml)


def update_key(key, default_value, settings_dict, parent_key=None):
    value = settings_dict.get(key, default_value)
    if value != default_value:
        if parent_key:
            print(f"Changed settings parameter for {key} under {parent_key} : {value} ")
        else:
            print(f"Changed settings parameter for {key} : {value} ")
    return value


def update_settings(config_dict: dict):
    # pylint: disable=too-many-branches, too-many-statements
    settings_dict = config_dict["settings"]

    valid_keys = [
        "grid_size",
        "field",
        "response",
        "observation",
        "localisation",
        "optional",
    ]
    for key in settings_dict:
        if key not in valid_keys:
            raise KeyError(f"Unknown keyword {key} in 'settings' ")

    key = "grid_size"
    grid_size_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = [
        "xsize",
        "ysize",
        "zsize",
        "polygon_file",
        "use_eclipse_grid_index_origo",
    ]
    if grid_size_dict:
        err_msg = []
        for sub_key in grid_size_dict:
            if sub_key not in valid_keys:
                err_msg.append(f"  {sub_key}")
        if len(err_msg) > 0:
            print(f"Unknown keywords in config file under keyword {key}: ")
            for text in err_msg:
                print(text)
            raise KeyError("Unknown keywords")

        grid_size_object = settings.grid_size
        grid_size_object.xsize = update_key(
            "xsize", grid_size_object.xsize, grid_size_dict, key
        )
        grid_size_object.ysize = update_key(
            "ysize", grid_size_object.ysize, grid_size_dict, key
        )
        grid_size_object.zsize = update_key(
            "zsize", grid_size_object.zsize, grid_size_dict, key
        )
        grid_size_object.polygon_file = update_key(
            "polygon_file", grid_size_object.polygon_file, grid_size_dict, key
        )
        grid_size_object.use_eclipse_grid_index_origo = update_key(
            "use_eclipse_grid_index_origo",
            grid_size_object.use_eclipse_grid_index_origo,
            grid_size_dict,
            key,
        )

    key = "field"
    field_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = [
        "name",
        "algorithm",
        "file_format",
        "initial_file_name",
        "updated_file_name",
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
        field.initial_file_name = update_key(
            "initial_file_name", field.initial_file_name, field_dict, key
        )
        field.updated_file_name = update_key(
            "updated_file_name", field.updated_file_name, field_dict, key
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
        "gen_data_file_name",
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
        response.gen_data_file_name = update_key(
            "gen_data_file_name", response.gen_data_file_name, response_dict, key
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
        "selected_grid_cells",
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
        observation.selected_grid_cells = update_key(
            "selected_grid_cells", observation.selected_grid_cells, obs_dict, key
        )

    key = "localisation"
    local_dict = settings_dict[key] if key in settings_dict else None
    valid_keys = ["method"]

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


def generate_field_and_upscale(real_number):
    seed_file_name = settings.field.seed_file
    relative_std = settings.field.trend_relstd
    use_trend = settings.field.trend_use
    algorithm_method = settings.field.algorithm
    start_seed = get_seed(seed_file_name, real_number)
    if algorithm_method == "gstools":
        print(f"Use algorithm: {algorithm_method}")
        residual_field = simulate_field_using_gstools(start_seed)
    else:
        print("Use algorithm: gaussianfft")
        residual_field = simulate_field(start_seed)
    if use_trend == 1:
        trend_field = trend()
        field3D = trend_field + relative_std * residual_field

    else:
        field3D = residual_field

    # Write field parameter for fine scale grid
    field_object = export_field(field3D)
    field_values = field_object.values

    # Calculate upscaled values for selected coarse grid cells
    upscaled_values = upscaling(
        field_values,
        iteration=0,
    )
    return upscaled_values


def get_seed(seed_file_name, r_number):
    with open(seed_file_name, "r", encoding="utf8") as file:
        lines = file.readlines()
        try:
            seed_value = int(lines[r_number - 1])
        except IndexError as exc:
            raise IOError("Seed value not found for realization {r_number}  ") from exc
        except ValueError as exc:
            raise IOError(
                "Invalid seed value in file for realization{r_number}"
            ) from exc
    return seed_value


def upscaling(field_values, iteration=0):
    """
    Calculate upscaled values and optionally write upscaled values to file.
    Return upscaled values
    """
    response_function_name = settings.response.response_function
    file_format = settings.response.file_format
    upscaled_field_name = settings.response.name
    write_upscaled_field = settings.response.write_upscaled_field
    NX, NY, NZ = settings.response.grid_dimension
    upscaled_values = np.zeros((NX, NY, NZ), dtype=np.float32, order="F")
    upscaled_values[:, :, :] = -999

    if response_function_name == "average":
        upscaled_values = upscale_average(
            field_values,
            upscaled_values,
        )

    if write_upscaled_field:
        upscaled_file_name = settings.response.upscaled_file_name
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
        nobs = get_nobs()
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(
                obs_number, nobs, selected_cell_index_list
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
                istart = mx * Iindx
                iend = istart + mx
                jstart = my * Jindx
                jend = jstart + my
                kstart = mz * Kindx
                kend = kstart + mz
                sum_val = 0.0
                for k in range(kstart, kend):
                    for j in range(jstart, jend):
                        for i in range(istart, iend):
                            sum_val += field_values[i, j, k]
                upscaled_values[Iindx, Jindx, Kindx] = sum_val / (mx * my * mz)
    return upscaled_values


def trend():
    """
    Return 3D numpy array with values following a linear trend
    scaled to take values between 0 and 1.
    """
    nx, ny, nz = settings.field.grid_dimension
    xsize = settings.grid_size.xsize
    ysize = settings.grid_size.ysize
    a, b = settings.field.trend_params

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


def simulate_field(start_seed):
    # pylint: disable=no-member,import-outside-toplevel
    # This function will not be available untill gaussianfft is available on python 3.10
    # import gaussianfft as sim  # isort: skip
    # dummy code to avoid pylint complaining in github actions
    sim = None

    variogram_name = settings.field.variogram
    corr_ranges = settings.field.correlation_range
    azimuth = settings.field.correlation_azimuth
    dip = settings.field.correlation_dip
    alpha = settings.field.correlation_exponent
    nx, ny, nz = settings.field.grid_dimension
    xsize = settings.grid_size.xsize
    ysize = settings.grid_size.ysize
    zsize = settings.grid_size.zsize

    xrange = corr_ranges[0]
    yrange = corr_ranges[1]
    zrange = corr_ranges[2]

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
    if settings.grid_size.use_eclipse_grid_index_origo:
        field_c_order = np.ma.zeros((nx, ny, nz), dtype=np.float32)
        j_indices = -np.arange(ny) + ny - 1
        # Flip j index and use c-order
        field_c_order[:, j_indices, :] = field_sim[:, :, :]
        return field_c_order
    # Change to C-order
    field_c_order = np.ma.zeros((nx, ny, nz), dtype=np.float32)
    field_c_order[:, :, :] = field_sim[:, :, :]
    return field_c_order


def simulate_field_using_gstools(start_seed):
    # pylint: disable=no-member,

    variogram_name = settings.field.variogram
    corr_ranges = settings.field.correlation_range
    azimuth = settings.field.correlation_azimuth
    xrange = corr_ranges[0]
    yrange = corr_ranges[1]
    zrange = corr_ranges[2]

    nx, ny, nz = settings.field.grid_dimension
    xsize = settings.grid_size.xsize
    ysize = settings.grid_size.ysize
    zsize = settings.grid_size.zsize

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    x = np.arange(0.5 * dx, xsize, dx)
    y = np.arange(0.5 * dy, ysize, dy)
    z = np.arange(0.5 * dz, zsize, dz)
    # Rescale factor is set to:
    #    sqrt(3.0) for gaussian correlation functions,
    #    3.0 for exponetial correlation function,
    #    pow(3.0, 1/alpha) for general exponential correlation function
    #    with exponent alpha
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
    # print(f"Field: {srf.field_names} ")
    # print(f"Field shape: {srf.field_shape} ")
    # print(f"Field type name: {srf.name}  ")
    # print(f"Field nugget: {srf.nugget} ")
    # print(f"Field opt arg:   {srf.opt_arg}")
    field = field_srf.reshape((nx, ny, nz))
    if settings.grid_size.use_eclipse_grid_index_origo:
        field_flip_j_index = np.ma.zeros((nx, ny, nz), dtype=np.float32)
        j_indices = -np.arange(ny) + ny - 1
        field_flip_j_index[:, j_indices, :] = field[:, :, :]
        return field_flip_j_index

    return field


def export_field(field3D):
    """
    Export initial realization of field to roff format
    Input field3D should be C-index order since xtgeo requires that
    """

    # nx, ny, nz = settings.field.grid_dimension
    field_name = settings.field.name
    field_file_name = settings.field.initial_file_name
    file_format = settings.field.file_format
    nx, ny, nz = settings.field.grid_dimension
    field_object = xtgeo.grid3d.GridProperty(
        ncol=nx, nrow=ny, nlay=nz, values=field3D, discrete=False, name=field_name
    )

    if file_format.upper() == "GRDECL":
        fullfilename = field_file_name + ".GRDECL"
        field_object.to_file(fullfilename, fformat="grdecl", dtype="float32")
    elif file_format.upper() == "ROFF":
        fullfilename = field_file_name + ".roff"
        field_object.to_file(fullfilename, fformat="roff")
    else:
        raise IOError(f"Unknown file format for fields: {file_format} ")
    print(f"Write field file: {fullfilename}  ")
    return field_object


def read_field_from_file():
    """
    Read field from roff formatted file and return xtgeo property object
    """

    input_file_name = settings.field.updated_file_name
    name = settings.field.name
    file_format = settings.field.file_format
    if file_format.upper() == "GRDECL":
        grid = xtgeo.grid_from_file(settings.field.grid_file_name, fformat="egrid")
        fullfilename = input_file_name + ".GRDECL"
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


def read_obs_field_from_file():
    """
    Read field parameter containing parameter with observed values
    for selected grid cells
    """

    file_format = settings.response.file_format
    pred_obs_file_name = settings.observation.reference_param_file
    if file_format.upper() == "ROFF":
        fullfilename = pred_obs_file_name + ".roff"
        obs_field_object = xtgeo.gridproperty_from_file(fullfilename, fformat="roff")
    elif file_format.upper() == "GRDECL":
        grid = xtgeo.grid_from_file(settings.response.grid_file_name, fformat="egrid")
        fullfilename = pred_obs_file_name + ".GRDECL"
        obs_field_object = xtgeo.gridproperty_from_file(
            fullfilename,
            fformat="grdecl",
            grid=grid,
            name=settings.observation.reference_field_name,
        )
    else:
        raise IOError(f"Unknown file format:{file_format} ")
    return obs_field_object


def read_upscaled_field_from_file(iteration):
    """
    Read upscaled field parameter either from initial ensemble or updated ensemble.
    Return xtgeo property object
    """

    input_file_name = settings.response.upscaled_file_name
    file_format = settings.response.file_format
    upscaled_field_name = settings.response.name
    if iteration == 0:
        filename = "init_files/" + input_file_name
    else:
        filename = input_file_name
    if file_format.upper() == "ROFF":
        fullfilename = filename + ".roff"
        field_object = xtgeo.gridproperty_from_file(fullfilename, fformat="roff")
    elif file_format.upper() == "GRDECL":
        grid = xtgeo.grid_from_file(settings.response.grid_file_name, fformat="egrid")
        fullfilename = filename + ".GRDECL"
        field_object = xtgeo.gridproperty_from_file(
            fullfilename, fformat="grdecl", grid=grid, name=upscaled_field_name
        )
    else:
        raise IOError(f"Unknown file format:{file_format} ")
    return field_object


def write_obs_pred_diff_field(upscaled_field_object, observation_field_object):
    """
    Get xtgeo property objects for predicted values for observables
    and observation values.
    Write file with difference as roff formatted file.
    """
    nx, ny, nz = upscaled_field_object.dimensions
    values_diff = upscaled_field_object.values - observation_field_object.values
    file_format = settings.response.file_format

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
        fullfilename = filename + ".GRDECL"
        diff_object.to_file(fullfilename, fformat="grdecl")
    else:
        raise IOError(f"Unknown file format: {file_format} ")
    print(
        "Write field with difference between upscaled reference "
        "field from which observations are extracted and "
        f"and prediction from current realization: {fullfilename}  "
    )


def get_cell_indices(obs_number, nobs, cell_indx_list):
    if nobs == 1:
        Iindx = cell_indx_list[0] - 1
        Jindx = cell_indx_list[1] - 1
        Kindx = cell_indx_list[2] - 1
    else:
        Iindx = cell_indx_list[obs_number][0] - 1
        Jindx = cell_indx_list[obs_number][1] - 1
        Kindx = cell_indx_list[obs_number][2] - 1

    return (Iindx, Jindx, Kindx)


def get_nobs():
    """
    Check if cell_indx_list is a single tuple (i,j,k) or
    a list of tuples  of type (i,j,k).
    Return number of cell indices found in list
    """
    cell_indx_list = settings.observation.selected_grid_cells
    is_list_of_ints = all(isinstance(indx, int) for indx in cell_indx_list)
    if is_list_of_ints:
        nobs = 1
    else:
        # list of tuples (i,j,k)
        nobs = len(cell_indx_list)
    return nobs
