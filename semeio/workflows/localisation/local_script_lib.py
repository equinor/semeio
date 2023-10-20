# pylint: disable=attribute-defined-outside-init,invalid-name
# pxylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-lines
import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Tuple, Optional

import cwrap
import numpy as np
import yaml
from ert.analysis.row_scaling import RowScaling
from ert.config import Field, GenDataConfig, GenKwConfig, SurfaceConfig
from ert.field_utils import save_field
from ert.field_utils.field_file_format import FieldFileFormat
from numpy import ma
from resdata.geometry import Surface
from resdata.grid.rd_grid import Grid
from resdata.rd_type import ResDataType
from resdata.resfile import Resdata3DKW

from semeio.workflows.localisation.localisation_debug_settings import (
    LogLevel,
    debug_print,
)


@dataclass
class Parameter:
    name: str
    parameters: List = field(default_factory=list)

    def to_list(self):
        if self.parameters:
            return [f"{self.name}:{parameter}" for parameter in self.parameters]
        return [f"{self.name}"]

    def to_dict(self):
        return {self.name: self.parameters}


@dataclass
class Parameters:
    parameters: List[Parameter] = field(default_factory=list)

    def append(self, new):
        self.parameters.append(new)

    def to_list(self):
        result = []
        for parameter in self.parameters:
            result.extend(parameter.to_list())
        return result

    def to_dict(self):
        result = {}
        for parameter in self.parameters:
            if parameter.name in result:
                raise ValueError(f"Duplicate parameters found: {parameter.name}")
            result.update(parameter.to_dict())
        return result

    @classmethod
    def from_list(cls, input_list):
        result = defaultdict(list)
        for item in input_list:
            words = item.split(":")
            if len(words) == 1:
                name = words[0]
                parameters = None
            elif len(words) == 2:
                name, parameters = words
            else:
                raise ValueError(f"Too many : in {item}")
            if name in result:
                if not parameters:
                    raise ValueError(
                        f"Inconsistent parameters, found {name} in "
                        f"{dict(result)}, but did not find parameters"
                    )
                if not result[name]:
                    raise ValueError(
                        f"Inconsistent parameters, found {name} in {dict(result)} but "
                        f"did not expect parameters, found {parameters}"
                    )
            if parameters:
                result[name].append(parameters)
            else:
                result[name] = []
        return cls([Parameter(key, val) for key, val in result.items()])


@dataclass
class Decay:
    obs_pos: list
    main_range: float
    perp_range: float
    azimuth: float
    grid: Grid

    def __call__(self, data_index):
        # Default behavior of Decay when called as a function
        # This is a placeholder; you can define a more meaningful default behavior
        return 1.0

    def __post_init__(self):
        angle = (90.0 - self.azimuth) * math.pi / 180.0
        self.cosangle = math.cos(angle)
        self.sinangle = math.sin(angle)

    def get_dx_dy(self, data_index):
        try:
            # Assume the grid is 3D Grid
            x, y, _ = self.grid.get_xyz(active_index=data_index)
        except AttributeError:
            # Assume the grid is a 2D Surface grid
            # pylint: disable=no-member
            x, y = self.grid.getXY(data_index)
        x_unrotated = x - self.obs_pos[0]
        y_unrotated = y - self.obs_pos[1]

        dx = (
            x_unrotated * self.cosangle + y_unrotated * self.sinangle
        ) / self.main_range
        dy = (
            -x_unrotated * self.sinangle + y_unrotated * self.cosangle
        ) / self.perp_range
        return dx, dy

    def norm_dist_square(self, data_index):
        dx, dy = self.get_dx_dy(data_index)
        d2 = dx**2 + dy**2
        return d2


@dataclass
class GaussianDecay(Decay):
    cutoff: bool

    def __call__(self, data_index: List[int]) -> float:
        d2 = super().norm_dist_square(data_index)
        if self.cutoff and d2 > 1.0:
            return 0.0
        exp_arg = -3.0 * d2
        return math.exp(exp_arg)


@dataclass
class ConstGaussianDecay(Decay):
    normalised_tapering_range: float
    cutoff: bool

    def __call__(self, data_index):
        d2 = super().norm_dist_square(data_index)
        d = math.sqrt(d2)
        if d <= 1.0:
            return 1.0
        if self.cutoff and d > self.normalised_tapering_range:
            return 0.0

        distance_from_inner_ellipse = (d - 1) / (self.normalised_tapering_range - 1)
        exp_arg = -3 * distance_from_inner_ellipse**2
        return math.exp(exp_arg)


@dataclass
class ExponentialDecay(Decay):
    cutoff: bool

    def __call__(self, data_index):
        d2 = super().norm_dist_square(data_index)
        d = math.sqrt(d2)
        if self.cutoff and d > 1.0:
            return 0.0
        exp_arg = -3.0 * d
        return math.exp(exp_arg)


@dataclass
class ConstExponentialDecay(Decay):
    normalised_tapering_range: float
    cutoff: bool

    def __call__(self, data_index):
        d2 = super().norm_dist_square(data_index)
        d = math.sqrt(d2)
        if d <= 1.0:
            return 1.0
        if self.cutoff and d > self.normalised_tapering_range:
            return 0.0

        distance_from_inner_ellipse = (d - 1) / (self.normalised_tapering_range - 1)
        exp_arg = -3 * distance_from_inner_ellipse
        return math.exp(exp_arg)


@dataclass
class ConstantScalingFactor:
    value: float

    def __call__(self, _):
        return self.value


class ScalingValues:
    scaling_param_number = 1
    corr_name = None

    @classmethod
    def initialize(cls):
        cls.scaling_param_number = 1
        cls.corr_name = None

    @classmethod
    def write_qc_parameter_field(
        cls,
        node_name,
        corr_name,
        field_scale,
        grid,
        param_for_field,
        log_level=LogLevel.OFF,
    ):
        # pylint: disable=too-many-arguments
        if param_for_field is None or field_scale is None:
            return

        # Write scaling parameter  once per corr_name
        if corr_name == cls.corr_name:
            return

        cls.corr_name = corr_name

        # Need a parameter name <= 8 character long for GRDECL format
        scaling_kw_name = "S_" + str(cls.scaling_param_number)
        filename = cls.corr_name + "_" + node_name + "_" + scaling_kw_name + ".roff"

        scaling_values = np.reshape(
            param_for_field, (grid.getNX(), grid.getNY(), grid.getNZ()), "C"
        )
        save_field(scaling_values, scaling_kw_name, filename, FieldFileFormat.ROFF)

        print(
            "Write calculated scaling factor  with name: "
            f"{scaling_kw_name} to file: {filename}"
        )
        debug_print(
            f"Write calculated scaling factor with name: "
            f"{scaling_kw_name} to file: {filename}",
            LogLevel.LEVEL3,
            log_level,
        )

        # Increase parameter number to define unique parameter name
        cls.scaling_param_number = cls.scaling_param_number + 1

    @classmethod
    def write_qc_parameter_surface(
        # pylint: disable=too-many-locals
        cls,
        node_name: str,
        corr_name: str,
        surface_scale: bool,
        reference_surface_file: str,
        param_for_surface: ma.MaskedArray,
        log_level: LogLevel = LogLevel.OFF,
    ) -> None:
        # pylint: disable=too-many-arguments

        if param_for_surface is None or surface_scale is None:
            return

        # Write scaling parameter  once per corr_name
        if corr_name == cls.corr_name:
            return

        cls.corr_name = corr_name

        scaling_surface_name = str(cls.scaling_param_number)
        filename = cls.corr_name + "_" + node_name + "_map.irap"

        # Surface object is created from the known reference surface to get
        # surface attributes. The param_for_surface array is c-order,
        # but need f-order for surface to write with the libecl object
        qc_surface = Surface(reference_surface_file)
        nx = qc_surface.getNX()
        ny = qc_surface.getNY()
        for i in range(nx):
            for j in range(ny):
                c_indx = j + i * ny
                f_indx = i + j * nx
                qc_surface[f_indx] = param_for_surface[c_indx]
        qc_surface.write(filename)
        print(
            "Write calculated scaling factor for SURFACE parameter "
            f"in {cls.corr_name} to file: {filename}"
        )
        debug_print(
            f"Write calculated scaling factor with name: "
            f"{scaling_surface_name} to file: {filename}",
            LogLevel.LEVEL3,
            log_level,
        )

        # Increase parameter number to define unique parameter name
        cls.scaling_param_number = cls.scaling_param_number + 1


def write_qc_parameter_field(
    node_name: str,
    corr_name_current: str,
    corr_name: str,
    scaling_param_number: int,
    field_scale: bool,
    grid: Grid,
    param_for_field: ma.MaskedArray,
    log_level: LogLevel = LogLevel.OFF,
) -> Tuple[str, int]:
    # pylint: disable=too-many-arguments
    if param_for_field is None or field_scale is None:
        return corr_name_current, scaling_param_number

    # Write scaling parameter  once per corr_name
    if corr_name == corr_name_current:
        return corr_name_current, scaling_param_number

    # Need a parameter name <= 8 character long for GRDECL format
    scaling_kw_name = "S_" + str(scaling_param_number)
    filename = corr_name + "_" + node_name + "_" + scaling_kw_name + ".roff"

    scaling_values = np.reshape(
        param_for_field, (grid.getNX(), grid.getNY(), grid.getNZ()), "C"
    )
    save_field(scaling_values, scaling_kw_name, filename, FieldFileFormat.ROFF)

    print(
        "Write calculated scaling factor  with name: "
        f"{scaling_kw_name} to file: {filename}"
    )
    debug_print(
        f"Write calculated scaling factor with name: "
        f"{scaling_kw_name} to file: {filename}",
        LogLevel.LEVEL3,
        log_level,
    )

    # Increase parameter number to define unique parameter name
    scaling_param_number = scaling_param_number + 1
    return corr_name, scaling_param_number


def write_qc_parameter_surface(
    # pylint: disable=too-many-locals
    node_name: str,
    corr_name_current: str,
    corr_name: str,
    scaling_param_number: int,
    surface_scale: bool,
    reference_surface_file: str,
    param_for_surface: ma.MaskedArray,
    log_level: LogLevel = LogLevel.OFF,
) -> Tuple[str, int]:
    # pylint: disable=too-many-arguments

    if param_for_surface is None or surface_scale is None:
        return corr_name_current, scaling_param_number

    # Write scaling parameter  once per corr_name
    if corr_name == corr_name_current:
        return corr_name_current, scaling_param_number

    scaling_surface_name = str(scaling_param_number)
    filename = corr_name + "_" + node_name + "_map.irap"

    # Surface object is created from the known reference surface to get
    # surface attributes. The param_for_surface array is c-order,
    # but need f-order for surface to write with the libecl object
    qc_surface = Surface(reference_surface_file)
    nx = qc_surface.getNX()
    ny = qc_surface.getNY()
    for i in range(nx):
        for j in range(ny):
            c_indx = j + i * ny
            f_indx = i + j * nx
            qc_surface[f_indx] = param_for_surface[c_indx]
    qc_surface.write(filename)
    print(
        "Write calculated scaling factor for SURFACE parameter "
        f"in {corr_name} to file: {filename}"
    )
    debug_print(
        f"Write calculated scaling factor with name: "
        f"{scaling_surface_name} to file: {filename}",
        LogLevel.LEVEL3,
        log_level,
    )

    # Increase parameter number to define unique parameter name
    scaling_param_number = scaling_param_number + 1
    return corr_name, scaling_param_number


def get_param_from_ert(ens_config):
    new_params = Parameters()
    for key in ens_config.parameters:
        node = ens_config.getNode(key)
        my_param = Parameter(key)
        new_params.append(my_param)
        if isinstance(node, GenKwConfig):
            my_param.parameters = node.getKeyWords()
    return new_params


def read_localisation_config(args):
    if len(args) == 1:
        specification_file_name = args[0]
    else:
        raise ValueError(f"Expecting a single argument. Got {args} arguments.")

    print(f"\nDefine localisation setup using config file: {specification_file_name}")
    logging.info(
        "\nDefine localisation setup using config file: %s", specification_file_name
    )
    with open(specification_file_name, "r", encoding="utf-8") as yml_file:
        localisation_yml = yaml.safe_load(yml_file)
    return localisation_yml


def activate_gen_kw_param(
    node_name: str,
    param_list: List[str],
    ert_param_dict: Dict[str, List[str]],
    log_level: LogLevel = LogLevel.OFF,
) -> List[int]:
    """
    Activate the selected parameters for the specified node.
    The param_list contains the list of parameters defined in GEN_KW
    for this node to be activated.
    """
    debug_print("Set active parameters", LogLevel.LEVEL2, log_level)
    all_params = ert_param_dict[node_name]
    index_list = []
    for param_name in param_list:
        index = all_params.index(param_name)
        if index is not None:
            debug_print(
                f"Active parameter: {param_name}  index: {index}",
                LogLevel.LEVEL3,
                log_level,
            )
            index_list.append(index)
    return index_list


def build_decay_object(
    method: str,
    ref_pos: List[float],
    main_range: float,
    perp_range: float,
    azimuth: float,
    grid: object,
    use_cutoff: bool,
    tapering_range: float = 1.5,
) -> Decay:
    # pylint: disable=too-many-arguments
    decay_obj: Union[
        GaussianDecay, ExponentialDecay, ConstGaussianDecay, ConstExponentialDecay
    ]
    if method == "gaussian_decay":
        decay_obj = GaussianDecay(
            ref_pos,
            main_range,
            perp_range,
            azimuth,
            grid,
            use_cutoff,
        )
    elif method == "exponential_decay":
        decay_obj = ExponentialDecay(
            ref_pos,
            main_range,
            perp_range,
            azimuth,
            grid,
            use_cutoff,
        )
    elif method == "const_gaussian_decay":
        decay_obj = ConstGaussianDecay(
            ref_pos,
            main_range,
            perp_range,
            azimuth,
            grid,
            tapering_range,
            use_cutoff,
        )
    elif method == "const_exponential_decay":
        decay_obj = ConstExponentialDecay(
            ref_pos,
            main_range,
            perp_range,
            azimuth,
            grid,
            tapering_range,
            use_cutoff,
        )
    else:
        _valid_methods = [
            "gaussian_decay",
            "exponential_decay",
            "const_gaussian_decay",
            "const_exponential_decay",
        ]
        raise NotImplementedError(
            f"The only allowed methods for function 'apply_decay' are: {_valid_methods}"
        )
    return decay_obj


def calculate_scaling_vector_fields(
    grid: object, decay_obj: Union[Decay, ConstantScalingFactor]
) -> ma.MaskedArray:
    assert isinstance(grid, Grid)
    nx = grid.getNX()
    ny = grid.getNY()
    nz = grid.getNZ()
    # Set 0 as initial value for scaling everywhere
    scaling_vector = np.ma.zeros(nx * ny * nz, dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # The grid uses fortran order and save only active cell values
                active_index = grid.get_active_index(ijk=(i, j, k))
                if active_index >= 0:
                    # active grid cell use c-index order
                    c_indx = k + j * nz + i * nz * ny
                    scaling_vector[c_indx] = decay_obj(active_index)

    return scaling_vector


def calculate_scaling_vector_surface(
    grid: object, decay_obj: Union[ConstantScalingFactor, Decay]
) -> ma.MaskedArray:
    assert isinstance(grid, Surface)
    nx = grid.getNX()
    ny = grid.getNY()

    # Set 0 as initial value for scaling everywhere
    scaling_vector = np.ma.zeros(nx * ny, dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            index = i + j * nx
            # Convert to c-order
            c_indx = j + i * ny
            scaling_vector[c_indx] = decay_obj(index)
    return scaling_vector


def apply_decay(
    method: str,
    row_scaling: RowScaling,
    grid: Grid,
    ref_pos: list,
    main_range: float,
    perp_range: float,
    azimuth: float,
    use_cutoff: bool = False,
    tapering_range: float = 1.5,
    calculate_qc_parameter: bool = False,
) -> Tuple[Optional[ma.MaskedArray], Optional[ma.MaskedArray]]:
    # pylint: disable=too-many-arguments,too-many-locals
    """
    Calculates the scaling factor, assign it to ERT instance by row_scaling
    and returns a full sized grid parameter with scaling factors for active
    grid cells and 0 elsewhere to be used for QC purpose.
    """
    decay_obj = build_decay_object(
        method,
        ref_pos,
        main_range,
        perp_range,
        azimuth,
        grid,
        use_cutoff,
        tapering_range,
    )
    if isinstance(grid, Grid):
        scaling_vector_for_fields = calculate_scaling_vector_fields(grid, decay_obj)
        row_scaling.assign_vector(scaling_vector_for_fields)

    elif isinstance(grid, Surface):
        scaling_vector_for_surface = calculate_scaling_vector_surface(grid, decay_obj)
        row_scaling.assign_vector(scaling_vector_for_surface)
    else:
        raise ValueError("No grid or surface object used in apply_decay")

    # Return field or surface scaling factor for QC purpose
    if calculate_qc_parameter:
        if isinstance(grid, Grid):
            return scaling_vector_for_fields, None
        if isinstance(grid, Surface):
            return None, scaling_vector_for_surface

    return None, None


def apply_constant(
    row_scaling: RowScaling,
    grid: Grid,
    value: float,
    log_level: LogLevel,
    calculate_qc_parameter: bool = False,
) -> Tuple[Optional[ma.MaskedArray], Optional[ma.MaskedArray]]:
    # pylint: disable=too-many-arguments,too-many-locals
    """
    Assign constant value to the scaling factor,
    assign it to ERT instance by row_scaling
    and returns a full sized grid parameter with scaling factors for active
    grid cells and 0 elsewhere to be used for QC purpose.
    """
    debug_print(f"Scaling factor is constant: {value}", LogLevel.LEVEL3, log_level)
    decay_obj = ConstantScalingFactor(value)
    if isinstance(grid, Grid):
        scaling_vector_field = calculate_scaling_vector_fields(grid, decay_obj)
        row_scaling.assign_vector(scaling_vector_field)
    elif isinstance(grid, Surface):
        scaling_vector_surface = calculate_scaling_vector_surface(grid, decay_obj)
        row_scaling.assign_vector(scaling_vector_surface)

    if calculate_qc_parameter:
        if isinstance(grid, Grid):
            return scaling_vector_field, None
        if isinstance(grid, Surface):
            return None, scaling_vector_surface
    return None, None


def apply_from_file(
    row_scaling: RowScaling,
    grid: Grid,
    filename: str,
    param_name: str,
    log_level: LogLevel,
    calculate_qc_parameter: bool = False,
) -> Tuple[Optional[ma.MaskedArray], None]:
    # pylint: disable=too-many-arguments, too-many-locals
    debug_print(
        f"Read scaling factors as parameter {param_name}", LogLevel.LEVEL3, log_level
    )
    debug_print(f"File name:  {filename}", LogLevel.LEVEL3, log_level)
    with cwrap.open(filename, "r") as file:
        scaling_parameter = Resdata3DKW.read_grdecl(
            grid,
            file,
            param_name,
            strict=True,
            rd_type=ResDataType.RD_FLOAT,
        )
    nx = grid.getNX()
    ny = grid.getNY()
    nz = grid.getNZ()

    # Set 0 as initial value for scaling everywhere
    scaling_vector = np.ma.zeros(nx * ny * nz, dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # The grid uses fortran order and save only active cell values
                active_index = grid.get_active_index(ijk=(i, j, k))
                if active_index >= 0:
                    # active grid cell use c-index order
                    c_indx = k + j * nz + i * nz * ny
                    global_index = grid.global_index(active_index=active_index)
                    scaling_vector[c_indx] = scaling_parameter[global_index]
    row_scaling.assign_vector(scaling_vector)

    # Return field or surface scaling factor for QC purpose
    if calculate_qc_parameter:
        return scaling_vector, None
    return None, None


def active_region(region_parameter, user_defined_active_region_list):
    """
     Find all region parameter values matching any of the regions defined
    to be used in localisation and mask the unused values
    """
    active_region_values_used = ma.zeros(len(region_parameter), dtype=np.int32)
    active_region_values_used[:] = -9999
    for region_number in user_defined_active_region_list:
        found_values = region_parameter == region_number
        active_region_values_used[found_values] = region_number
    is_not_used = active_region_values_used == -9999
    active_region_values_used.mask = is_not_used
    return active_region_values_used


def define_look_up_index(user_defined_active_region_list, max_region_number):
    """
    Define an array taking region number as input and returning the index in the
    user define active region list. Is used for fast lookup of scaling parameter
    corresponding to the region number.
    """
    active_segment_array = np.array(user_defined_active_region_list)
    index_per_used_region = ma.zeros((max_region_number + 1), dtype=np.int32)
    index_values = np.arange(len(active_segment_array))
    index_per_used_region[active_segment_array[index_values]] = index_values
    return index_per_used_region


def calculate_scaling_factors_in_regions(
    region_parameter, active_segment_list, scaling_value_list
):
    min_region_number = region_parameter.min()
    max_region_number = region_parameter.max()

    # Get a list of region numbers that exists in region parameter
    regions_in_param = []
    for region_number in range(min_region_number, max_region_number + 1):
        has_region = region_parameter == region_number
        if has_region.any():
            regions_in_param.append(region_number)

    active_region_values_used = active_region(region_parameter, active_segment_list)
    index_per_used_region = define_look_up_index(active_segment_list, max_region_number)
    scaling_value_array = np.array(scaling_value_list)

    # Get selected (not masked) region values
    selected_grid_cells = np.logical_not(active_region_values_used.mask)
    selected_region_values = active_region_values_used[selected_grid_cells]

    # Look up scaling values for selected region values
    scaling_values_active = scaling_value_array[
        index_per_used_region[selected_region_values]
    ]

    # Create a full sized 3D parameter for scaling values
    # where all but the selected region values have 0 scaling value.
    scaling_values = np.ma.zeros(len(region_parameter), dtype=np.float32)
    scaling_values[selected_grid_cells] = scaling_values_active

    return scaling_values, active_region_values_used, regions_in_param


def smooth_parameter(
    grid: Grid,
    smooth_range_list: List[int],
    scaling_values: List[float],
    active_region_values_used: ma.MaskedArray,
) -> ma.MaskedArray:
    """
    Function taking as input a 3D parameter  scaling_values and calculates a new
    3D parameter scaling_values_smooth using local average within a rectangular window
    around the cell to be assigned the smoothed value. The smoothing window is
    defined by the two range parameters in smooth_range_list.
    They contain integer values >=0 and smooth_range_list = [0,0] means no smoothing.
    The input parameter active_region_values_used has non-negative integer values
    with region number for all grid cells containing values for the input 3D parameter
    scaling_values. All other grid cells are masked.
    The smoothing algorithm is defined such that only values not masked are used.
    If the  scaling_values contain constant values for each
    active region and e.g 0 for all inactive regions and for inactive grid cells,
    then the smoothing will only appear on the border between active regions.
    """
    # pylint: disable=too-many-locals
    nx = grid.get_nx()
    ny = grid.get_ny()
    nz = grid.get_nz()
    di = smooth_range_list[0]
    dj = smooth_range_list[1]
    scaling_values_smooth = np.ma.zeros(nx * ny * nz, dtype=np.float32)
    for k, j0, i0 in itertools.product(range(nz), range(ny), range(nx)):
        index0 = k + j0 * nz + i0 * nz * ny
        if active_region_values_used[index0] is not ma.masked:
            sumv = 0.0
            nval = 0
            ilow: int = max(0, i0 - di)
            ihigh: int = min(i0 + di + 1, nx)
            jlow: int = max(0, j0 - dj)
            jhigh: int = min(j0 + dj + 1, ny)
            for i in range(ilow, ihigh):
                for j in range(jlow, jhigh):
                    index = k + j * nz + i * nz * ny
                    if active_region_values_used[index] is not ma.masked:
                        # Only use values from grid cells that are active
                        # and from regions defined as active by the user.
                        v = scaling_values[index]
                        sumv += v
                        nval += 1
            if nval > 0:
                scaling_values_smooth[index0] = sumv / nval
    return scaling_values_smooth


def apply_segment(
    row_scaling: RowScaling,
    grid: Grid,
    region_param_dict: Dict[str, Any],
    active_segment_list: List[int],
    scaling_factor_list: List[float],
    smooth_range_list: List[int],
    corr_name: str,
    log_level: LogLevel = LogLevel.OFF,
    calculate_qc_parameter: bool = False,
) -> Tuple[Any, None]:
    # pylint: disable=too-many-arguments,too-many-locals
    """
    Purpose: Use region numbers and list of scaling factors per region to
                   create scaling factors per active .
                   Input dictionary with keyword which is correlation group name,
                   where values are numpy vector with region parameters
                   for each grid cell in ERTBOX grid.
                   A scaling factor is specified for each specified active region.
                   Optionally also a spatial smoothing of the scaling factors
                   can be done by specifying smooth ranges in number of
                   grid cells in I and J direction. If this is not specified,
                   no smoothing is done.
                   NOTE: Smoothing is done only between active segments,
                   and no smoothing between active segments and inactive
                   segments or inactive grid cells.
    """

    debug_print(f"Active segments: {active_segment_list}", LogLevel.LEVEL3, log_level)

    max_region_number_specified = max(active_segment_list)

    region_parameter = region_param_dict[corr_name]
    max_region_parameter = region_parameter.max()
    if max_region_parameter < max_region_number_specified:
        raise ValueError(
            "Specified an active region with number "
            f"{max_region_number_specified} which is larger \n"
            f"than max region parameter {max_region_parameter} for "
            f"correlation group {corr_name}."
        )

    (
        scaling_values,
        active_localisation_region,
        regions_in_param,
    ) = calculate_scaling_factors_in_regions(
        region_parameter,
        active_segment_list,
        scaling_factor_list,
    )
    if smooth_range_list is not None:
        scaling_values = smooth_parameter(
            grid, smooth_range_list, scaling_values, active_localisation_region
        )

    # Assign values to row_scaling object
    row_scaling.assign_vector(scaling_values)

    not_defined_in_region_param = []
    for n in active_segment_list:
        if n not in regions_in_param:
            not_defined_in_region_param.append(n)
    if len(not_defined_in_region_param) > 0:
        debug_print(
            f"Warning: The following region numbers are specified in \n"
            "                config file for correlation group "
            f"{corr_name}, \n"
            "                but not found in region parameter: "
            f"{not_defined_in_region_param}",
            LogLevel.LEVEL3,
            log_level,
        )
    if calculate_qc_parameter:
        return scaling_values, None
    return None, None


def read_region_files_for_all_correlation_groups(user_config, grid):
    # pylint: disable=too-many-nested-blocks,too-many-locals,invalid-name
    if grid is None:
        # No grid is defined. Not relevant to look for region files to read.
        return None

    region_param_dict = {}
    corr_name_dict = {}
    nx = grid.get_nx()
    ny = grid.get_ny()
    nz = grid.get_nz()
    for _, corr_spec in enumerate(user_config.correlations):
        region_param_dict[corr_spec.name] = None
        if corr_spec.field_scale is not None:
            if corr_spec.field_scale.method == "segment":
                filename = corr_spec.field_scale.segment_filename
                param_name = corr_spec.field_scale.param_name
                debug_print(
                    f"Use parameter: {param_name} from file: {filename} "
                    f"in {corr_spec.name}",
                    LogLevel.LEVEL2,
                    user_config.log_level,
                )

                if filename not in corr_name_dict:
                    # Read the file
                    with cwrap.open(filename, "r") as file:
                        region_parameter_read = Resdata3DKW.read_grdecl(
                            grid,
                            file,
                            param_name,
                            strict=True,
                            rd_type=ResDataType.RD_INT,
                        )
                    region_parameter = np.zeros(nx * ny * nz, dtype=np.int32)
                    not_active = np.zeros(nx * ny * nz, dtype=np.int32)
                    for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
                        # c-index order
                        index = k + j * nz + i * nz * ny
                        v = region_parameter_read[i, j, k]
                        region_parameter[index] = v
                        if grid.get_active_index(ijk=(i, j, k)) == -1:
                            not_active[index] = 1
                    region_parameter_masked = ma.masked_array(
                        region_parameter, mask=not_active
                    )
                    region_param_dict[corr_spec.name] = region_parameter_masked
                    corr_name_dict[filename] = corr_spec.name
                else:
                    # The region_parameter is already read for a previous
                    # correlation group. Re-use it instead of re-reading the file
                    existing_corr_name = corr_name_dict[filename]
                    region_param_dict[corr_spec.name] = region_param_dict[
                        existing_corr_name
                    ]
    return region_param_dict


def add_ministeps(
    user_config,
    ert_param_dict,
    ert_ensemble_config,
    grid_for_field,
):
    # pylint: disable=too-many-branches,too-many-statements
    # pylint: disable=too-many-nested-blocks,too-many-locals

    debug_print("Add all ministeps:", LogLevel.LEVEL1, user_config.log_level)
    number_of_scaling_parameter_of_type_field = 0
    corr_name_using_fields = None
    number_of_scaling_parameter_of_type_surface = 0
    corr_name_using_surface = None
    # ScalingValues.initialize()
    # Read all region files used in correlation groups,
    # but only once per unique region file.

    region_param_dict = read_region_files_for_all_correlation_groups(
        user_config, grid_for_field
    )
    update_steps = []
    for corr_spec in user_config.correlations:
        debug_print(
            f"Define ministep: {corr_spec.name}", LogLevel.LEVEL1, user_config.log_level
        )
        ministep_name = corr_spec.name
        update_step = defaultdict(list)
        update_step["name"] = corr_spec.name
        obs_list = corr_spec.obs_group.result_items
        param_dict = Parameters.from_list(corr_spec.param_group.result_items).to_dict()

        # Setup model parameter group
        for node_name, param_list in param_dict.items():
            node = ert_ensemble_config.getNode(node_name)
            impl_type = type(node).__name__
            debug_print(
                f"Add node: {node_name} of type: {impl_type}",
                LogLevel.LEVEL2,
                user_config.log_level,
            )
            if isinstance(node, GenKwConfig):
                index_list = activate_gen_kw_param(
                    node_name,
                    param_list,
                    ert_param_dict,
                    user_config.log_level,
                )
                update_step["parameters"].append([node_name, index_list])
            elif isinstance(node, Field):
                assert grid_for_field is not None
                _decay_methods_group1 = ["gaussian_decay", "exponential_decay"]
                _decay_methods_group2 = [
                    "const_gaussian_decay",
                    "const_exponential_decay",
                ]
                _decay_methods_all = _decay_methods_group1 + _decay_methods_group2
                if corr_spec.field_scale is not None:
                    debug_print(
                        "Scale field parameter correlations using method: "
                        f"{corr_spec.field_scale.method}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
                    row_scaling = RowScaling()
                    param_for_field = None
                    if corr_spec.field_scale.method in _decay_methods_all:
                        ref_pos = corr_spec.field_scale.ref_point
                        main_range = corr_spec.field_scale.main_range
                        perp_range = corr_spec.field_scale.perp_range
                        azimuth = corr_spec.field_scale.azimuth
                        use_cutoff = corr_spec.field_scale.cutoff
                        tapering_range = None
                        if corr_spec.field_scale.method in _decay_methods_group2:
                            tapering_range = (
                                corr_spec.field_scale.normalised_tapering_range
                            )
                        check_if_ref_point_in_grid(
                            ref_pos, grid_for_field, log_level=user_config.log_level
                        )
                        (param_for_field, _) = apply_decay(
                            corr_spec.field_scale.method,
                            row_scaling,
                            grid_for_field,
                            ref_pos,
                            main_range,
                            perp_range,
                            azimuth,
                            use_cutoff,
                            tapering_range,
                            user_config.write_scaling_factors,
                        )
                    elif corr_spec.field_scale.method == "constant":
                        (param_for_field, _) = apply_constant(
                            row_scaling,
                            grid_for_field,
                            corr_spec.field_scale.value,
                            user_config.log_level,
                            user_config.write_scaling_factors,
                        )
                    elif corr_spec.field_scale.method == "from_file":
                        (param_for_field, _) = apply_from_file(
                            row_scaling,
                            grid_for_field,
                            corr_spec.field_scale.filename,
                            corr_spec.field_scale.param_name,
                            user_config.log_level,
                            user_config.write_scaling_factors,
                        )

                    elif corr_spec.field_scale.method == "segment":
                        (param_for_field, _) = apply_segment(
                            row_scaling,
                            grid_for_field,
                            region_param_dict,
                            corr_spec.field_scale.active_segments,
                            corr_spec.field_scale.scalingfactors,
                            corr_spec.field_scale.smooth_ranges,
                            corr_spec.name,
                            user_config.log_level,
                            user_config.write_scaling_factors,
                        )
                    else:
                        logging.error(
                            "Scaling method: %s is not implemented.",
                            corr_spec.field_scale.method,
                        )
                        raise ValueError(
                            f"Scaling method: {corr_spec.field_scale.method} "
                            "is not implemented"
                        )

                    if user_config.write_scaling_factors:
                        #                        ScalingValues.write_qc_parameter_field(
                        (
                            corr_name_using_fields,
                            number_of_scaling_parameter_of_type_field,
                        ) = write_qc_parameter_field(
                            node_name,
                            corr_name_using_fields,
                            corr_spec.name,
                            number_of_scaling_parameter_of_type_field,
                            corr_spec.field_scale,
                            grid_for_field,
                            param_for_field,
                            user_config.log_level,
                        )
                    update_step["row_scaling_parameters"].append(
                        [node_name, row_scaling]
                    )
                else:
                    # The keyword for how to specify scaling parameter for field
                    # is not used. Use default: scaling factor = 1 everywhere
                    row_scaling = RowScaling()
                    scaling_factor_default = 1.0
                    (param_for_field, _) = apply_constant(
                        row_scaling,
                        grid_for_field,
                        scaling_factor_default,
                        user_config.log_level,
                        user_config.write_scaling_factors,
                    )
                    update_step["row_scaling_parameters"].append(
                        [node_name, row_scaling]
                    )

                    debug_print(
                        f"No correlation scaling specified for node {node_name} "
                        f"in {ministep_name}. "
                        f"Use default scaling factor: {scaling_factor_default}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
            elif isinstance(node, GenDataConfig):
                debug_print(
                    f"Parameter {node_name} of type: {impl_type} "
                    f"in {ministep_name}",
                    LogLevel.LEVEL3,
                    user_config.log_level,
                )
            elif isinstance(node, SurfaceConfig):
                _decay_methods_surf_group1 = ["gaussian_decay", "exponential_decay"]
                _decay_methods_surf_group2 = [
                    "const_gaussian_decay",
                    "const_exponential_decay",
                ]
                _decay_methods_surf_all = (
                    _decay_methods_surf_group1 + _decay_methods_surf_group2
                )
                if corr_spec.surface_scale is not None:
                    surface_file = corr_spec.surface_scale.surface_file
                    debug_print(
                        f"Get surface size from: {surface_file}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
                    debug_print(
                        "Scale surface parameter correlations using method: "
                        f"{corr_spec.surface_scale.method}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )

                    surface = Surface(surface_file)
                    row_scaling = RowScaling()
                    if corr_spec.surface_scale.method in _decay_methods_surf_all:
                        ref_pos = corr_spec.surface_scale.ref_point
                        main_range = corr_spec.surface_scale.main_range
                        perp_range = corr_spec.surface_scale.perp_range
                        azimuth = corr_spec.surface_scale.azimuth
                        use_cutoff = corr_spec.surface_scale.cutoff
                        tapering_range = None
                        if corr_spec.surface_scale.method in _decay_methods_surf_group2:
                            tapering_range = (
                                corr_spec.surface_scale.normalised_tapering_range
                            )
                        (_, param_for_surface) = apply_decay(
                            corr_spec.surface_scale.method,
                            row_scaling,
                            surface,
                            ref_pos,
                            main_range,
                            perp_range,
                            azimuth,
                            use_cutoff,
                            tapering_range,
                            user_config.write_scaling_factors,
                        )
                    elif corr_spec.surface_scale.method == "constant":
                        (_, param_for_surface) = apply_constant(
                            row_scaling,
                            surface,
                            corr_spec.surface_scale.value,
                            user_config.log_level,
                            user_config.write_scaling_factors,
                        )
                    else:
                        logging.error(
                            "Scaling method: %s is not implemented.",
                            corr_spec.surface_scale.method,
                        )
                        raise ValueError(
                            f"Scaling method: {corr_spec.surface_scale.method} "
                            "is not implemented"
                        )

                    (
                        corr_name_using_surface,
                        number_of_scaling_parameter_of_type_surface,
                    ) = write_qc_parameter_surface(
                        # pylint: disable=too-many-locals
                        node_name,
                        corr_name_using_surface,
                        corr_spec.name,
                        number_of_scaling_parameter_of_type_surface,
                        corr_spec.surface_scale,
                        corr_spec.surface_scale.surface_file,
                        param_for_surface,
                        user_config.log_level,
                    )

                    update_step["row_scaling_parameters"].append(
                        [node_name, row_scaling]
                    )
                else:
                    debug_print(
                        "Surface parameter is specified, but no 'surface_scale' "
                        "keyword is specified. Require that 'surface_scale' "
                        "keyword is specified.",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
                    raise KeyError(
                        f" When using surface parameter {node_name} the keyword"
                        f" 'surface_scale' must be specified."
                    )
        # Setup observation group
        update_step["observations"] = obs_list
        debug_print(
            f"Observations in {ministep_name}:   {obs_list} ",
            LogLevel.LEVEL3,
            user_config.log_level,
        )

        update_steps.append(update_step)

    return update_steps


def check_if_ref_point_in_grid(ref_point, grid, log_level):
    try:
        (i_indx, j_indx) = grid.find_cell_xy(ref_point[0], ref_point[1], 0)
    except ValueError as err:
        raise ValueError(
            f"Reference point {ref_point} corresponds to undefined grid cell "
            f"or is outside the area defined by the grid {grid.get_name()}\n"
            "Check specification of reference point ",
            "and grid index origin of grid with field parameters.",
        ) from err

    debug_print(
        f"Reference point {ref_point} has grid indices: ({i_indx}, {j_indx})",
        LogLevel.LEVEL3,
        log_level,
    )
