# pylint: disable=W0201
import math
from collections import defaultdict
from typing import List

import yaml
import cwrap
import numpy as np
import logging

from ecl.util.geometry import Surface
from res.enkf.enums.ert_impl_type_enum import ErtImplType
from res.enkf.enums.enkf_var_type_enum import EnkfVarType

from dataclasses import dataclass, field

from semeio.workflows.localisation.localisation_debug_settings import (
    LogLevel,
    debug_print,
)


@dataclass
class Parameter:
    name: str
    type: str
    parameters: List = field(default_factory=list)

    def to_list(self):
        if self.parameters:
            return [f"{self.name}:{parameter}" for parameter in self.parameters]
        else:
            return [f"{self.name}"]

    def to_dict(self):
        return {self.name: self.parameters}


@dataclass
class Parameters:
    parameters: List[Parameter] = field(default_factory=list)

    def append(self, new):
        self.parameters.append(new)

    def to_list(self):
        # pylint: disable=E1133
        result = []
        for parameter in self.parameters:
            result.extend(parameter.to_list())
        return result

    def to_dict(self):
        # pylint: disable=E1133
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
        return cls([Parameter(key, "UNKNOWN", val) for key, val in result.items()])


def get_param_from_ert(ens_config):
    new_params = Parameters()
    keylist = ens_config.alloc_keylist()
    implementation_type_not_scalar = [
        ErtImplType.GEN_DATA,
        ErtImplType.FIELD,
        ErtImplType.SURFACE,
    ]
    for key in keylist:
        node = ens_config.getNode(key)
        impl_type = node.getImplementationType()
        if node.getVariableType() == EnkfVarType.PARAMETER:
            my_param = Parameter(key, impl_type)
            new_params.append(my_param)
            if impl_type == ErtImplType.GEN_KW:
                # Node contains scalar parameters defined by GEN_KW
                kw_config_model = node.getKeywordModelConfig()
                my_param.parameters = kw_config_model.getKeyWords().strings
            elif impl_type in implementation_type_not_scalar:
                # Node contains parameter from FIELD, SURFACE or GEN_PARAM
                # The parameters_for_node dict contains empty list for FIELD
                # and SURFACE.
                # The number of variables for a FIELD parameter is
                # defined by the grid in the GRID keyword.
                # The number of variables for a SURFACE parameter is
                # defined by the size of a surface object.
                # For GEN_PARAM the list contains the number of variables.
                # parameters_for_node[key] = []
                if impl_type == ErtImplType.GEN_DATA:
                    gen_data_config = node.getDataModelConfig()
                    data_size = gen_data_config.get_initial_size()
                    if data_size <= 0:
                        # Cannot here know if it will be used or not.
                        logging.warning(
                            "\nThe ERT config has defined parameter nodes of type\n"
                            "GEN_PARAM. If this is used in localisation, \n"
                            "the localisation workflow must be run AFTER initial \n"
                            "ensemble is created, but BEFORE first update is run.\n\n"
                        )

                    my_param.parameters = [str(item) for item in range(data_size)]

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


def active_index_for_parameter(node_name, param_name, ert_param_dict):
    # For parameters defined as scalar parameters (coming from GEN_KW)
    # the parameters for a node have a name. Get the index from the order
    # of these parameters.

    ert_param_list = ert_param_dict[node_name]
    if len(ert_param_list) > 0:
        index = -1
        for count, name in enumerate(ert_param_list):
            if name == param_name:
                index = count
                break
        assert index > -1
    return index


def activate_gen_kw_param(
    model_param_group, node_name, param_list, ert_param_dict, log_level=LogLevel.OFF
):
    """
    Activate the selected parameters for the specified node.
    The param_list contains the list of parameters defined in GEN_KW
    for this node to be activated.
    """
    active_param_list = model_param_group.getActiveList(node_name)
    debug_print("Set active parameters", LogLevel.LEVEL2, log_level)
    for param_name in param_list:
        index = active_index_for_parameter(node_name, param_name, ert_param_dict)
        if index is not None:
            debug_print(
                f"Active parameter: {param_name}  index: {index}",
                LogLevel.LEVEL3,
                log_level,
            )
            active_param_list.addActiveIndex(index)


def activate_gen_param(
    model_param_group, node_name, param_list, data_size, log_level=LogLevel.OFF
):
    """
    Activate the selected parameters for the specified node.
    The param_list contains a list of names that are integer numbers
    for the parameter indices to be activated for parameters belonging
    to the specified GEN_PARAM node.
    """
    active_param_list = model_param_group.getActiveList(node_name)
    for param_name in param_list:
        index = int(param_name)
        if index < 0 or index >= data_size:
            raise ValueError(
                f"Index for parameter in node {node_name} is "
                f"outside the interval [0,{data_size-1}]"
            )
        debug_print(f"Active parameter index: {index}", LogLevel.LEVEL3, log_level)
        active_param_list.addActiveIndex(index)


def activate_and_scale_correlation(
    grid,
    method,
    row_scaling,
    ref_pos,
    data_size,
    main_range,
    perp_range,
    azimuth,
    log_level=LogLevel.OFF,
):
    debug_print(
        f"Scale correlations using method: {method}",
        LogLevel.LEVEL3,
        log_level,
    )
    if method == "gaussian_decay":
        row_scaling.assign(
            data_size,
            GaussianDecay(ref_pos, main_range, perp_range, azimuth, grid),
        )
    elif method == "exponential_decay":
        row_scaling.assign(
            data_size,
            ExponentialDecay(ref_pos, main_range, perp_range, azimuth, grid),
        )
    else:
        logging.warning("Scaling method: %s is not implemented.", method)


def add_ministeps(
    user_config,
    ert_param_dict,
    ert_local_config,
    ert_ensemble_config,
    grid_for_field,
):
    # pylint: disable-msg=too-many-branches
    # pylint: disable-msg=R0915

    debug_print("Add all ministeps:", LogLevel.LEVEL1, user_config.log_level)
    ScalingValues.initialize()
    for count, corr_spec in enumerate(user_config.correlations):
        ministep_name = corr_spec.name
        ministep = ert_local_config.createMinistep(ministep_name)
        debug_print(
            f"Define ministep: {ministep_name}", LogLevel.LEVEL1, user_config.log_level
        )

        param_group_name = ministep_name + "_param_group"
        model_param_group = ert_local_config.createDataset(param_group_name)

        obs_group_name = ministep_name + "_obs_group"
        obs_group = ert_local_config.createObsdata(obs_group_name)

        obs_list = corr_spec.obs_group.result_items
        param_dict = Parameters.from_list(corr_spec.param_group.result_items).to_dict()

        # Setup model parameter group
        for node_name, param_list in param_dict.items():
            node = ert_ensemble_config.getNode(node_name)
            impl_type = node.getImplementationType()

            debug_print(
                f"Add node: {node_name} of type: {impl_type}",
                LogLevel.LEVEL2,
                user_config.log_level,
            )
            model_param_group.addNode(node_name)
            if impl_type == ErtImplType.GEN_KW:
                activate_gen_kw_param(
                    model_param_group,
                    node_name,
                    param_list,
                    ert_param_dict,
                    user_config.log_level,
                )
            elif impl_type == ErtImplType.FIELD:
                if corr_spec.field_scale is not None:
                    field_config = node.getFieldModelConfig()
                    check_if_ref_point_in_grid(corr_spec.ref_point, grid_for_field)
                    activate_and_scale_correlation(
                        grid_for_field,
                        corr_spec.field_scale.method,
                        model_param_group.row_scaling(node_name),
                        corr_spec.ref_point,
                        field_config.get_data_size(),
                        corr_spec.field_scale.main_range,
                        corr_spec.field_scale.perp_range,
                        corr_spec.field_scale.azimuth,
                        user_config.log_level,
                    )
                    if user_config.write_scaling_factors:
                        ScalingValues.write(
                            node_name, corr_spec, grid_for_field, user_config.log_level
                        )
                else:
                    debug_print(
                        f"No correlation scaling for node {node_name} "
                        f"in {ministep_name}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
            elif impl_type == ErtImplType.GEN_DATA:
                gen_data_config = node.getDataModelConfig()
                data_size = gen_data_config.get_initial_size()
                if data_size > 0:
                    activate_gen_param(
                        model_param_group,
                        node_name,
                        param_list,
                        data_size,
                        user_config.log_level,
                    )
                else:
                    debug_print(
                        f"Parameter {node_name} has data size: {data_size} "
                        f"in {ministep_name}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
            elif impl_type == ErtImplType.SURFACE:
                if corr_spec.surface_scale is not None:
                    surface_file = corr_spec.surface_scale.surface_file
                    debug_print(
                        f"Get surface size from: {surface_file}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )
                    surface = Surface(surface_file)
                    data_size = surface.getNX() * surface.getNY()
                    activate_and_scale_correlation(
                        surface,
                        corr_spec.surface_scale.method,
                        model_param_group.row_scaling(node_name),
                        corr_spec.ref_point,
                        data_size,
                        corr_spec.surface_scale.main_range,
                        corr_spec.surface_scale.perp_range,
                        corr_spec.surface_scale.azimuth,
                    )
                else:
                    debug_print(
                        f"No correlation scaling for node {node_name} "
                        f"in {ministep_name}",
                        LogLevel.LEVEL3,
                        user_config.log_level,
                    )

        # Setup observation group
        for obs_name in obs_list:
            debug_print(
                f"Add obs node: {obs_name}", LogLevel.LEVEL2, user_config.log_level
            )
            obs_group.addNode(obs_name)

        # Setup ministep
        debug_print(
            f"Attach {param_group_name} to ministep {ministep_name}",
            LogLevel.LEVEL1,
            user_config.log_level,
        )
        ministep.attachDataset(model_param_group)

        debug_print(
            f"Attach {obs_group_name} to ministep {ministep_name}",
            LogLevel.LEVEL1,
            user_config.log_level,
        )
        ministep.attachObsset(obs_group)

        debug_print(
            f"Add {ministep_name} to update step\n",
            LogLevel.LEVEL1,
            user_config.log_level,
        )
        ert_local_config.getUpdatestep().attachMinistep(ministep)


def clear_correlations(ert):
    local_config = ert.getLocalConfig()
    local_config.clear()


def check_if_ref_point_in_grid(ref_point, grid):
    try:
        grid.find_cell_xy(ref_point[0], ref_point[1], 0)
    except ValueError as err:
        raise ValueError(
            f"Reference point {ref_point} corresponds to undefined grid cell "
            f"or is outside the area defined by the grid {grid.get_name()}\n"
            "Check specification of reference point."
        ) from err


@dataclass
class Decay:
    obs_pos: list
    main_range: float
    perp_range: float
    azimuth: float
    grid: object

    def __post_init__(self):
        angle = (90.0 - self.azimuth) * math.pi / 180.0
        self.cosangle = math.cos(angle)
        self.sinangle = math.sin(angle)

    def get_dx_dy(self, data_index):
        try:
            x, y, _ = self.grid.get_xyz(active_index=data_index)
        except AttributeError:
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


class GaussianDecay(Decay):
    def __call__(self, data_index):
        dx, dy = super().get_dx_dy(data_index)
        exp_arg = -0.5 * (dx * dx + dy * dy)
        return math.exp(exp_arg)


class ExponentialDecay(Decay):
    def __call__(self, data_index):
        dx, dy = super().get_dx_dy(data_index)
        exp_arg = -0.5 * math.sqrt(dx * dx + dy * dy)
        return math.exp(exp_arg)


class ScalingValues:
    # pylint: disable=R0903
    scaling_param_number = 1
    corr_name = None

    @classmethod
    def initialize(cls):
        cls.scaling_param_number = 1
        cls.corr_name = None

    @classmethod
    def write(cls, node_name, corr_spec, grid_for_field, log_level):
        corr_name = corr_spec.name
        ref_point = corr_spec.ref_point
        grid = grid_for_field
        if corr_spec.field_scale is None:
            raise ValueError(
                "Scaling values for 3D parameters require "
                f"specification of keyword 'field_scale' for {cls.corr_name}"
            )
        method_name = corr_spec.field_scale.method

        if method_name == "gaussian_decay":
            decay_obj = GaussianDecay(
                ref_point,
                corr_spec.field_scale.main_range,
                corr_spec.field_scale.perp_range,
                corr_spec.field_scale.azimuth,
                grid,
            )
            scaling_values = cls._calculate_scaling_param(grid, decay_obj)
        elif method_name == "exponential_decay":
            decay_obj = ExponentialDecay(
                ref_point,
                corr_spec.field_scale.main_range,
                corr_spec.field_scale.perp_range,
                corr_spec.field_scale.azimuth,
                grid,
            )
            scaling_values = cls._calculate_scaling_param(grid, decay_obj)
        else:
            raise KeyError(f" Method name: {method_name} is not implemented")

        # Write scaling parameter  once per corr_name
        if corr_name != cls.corr_name:
            cls.corr_name = corr_name
            scaling_kw_name = "S_" + str(cls.scaling_param_number)
            scaling_kw = grid.create_kw(scaling_values, scaling_kw_name, False)
            filename = (
                cls.corr_name + "_" + node_name + "_" + scaling_kw_name + ".GRDECL"
            )
            debug_print(f"Write file: {filename}", LogLevel.LEVEL3, log_level)
            with cwrap.open(filename, "w") as file:
                grid.write_grdecl(scaling_kw, file)
            # Increase parameter number to define unique parameter name
            cls.scaling_param_number = cls.scaling_param_number + 1

    @classmethod
    def _calculate_scaling_param(cls, grid, decay_obj):
        nx = grid.getNX()
        ny = grid.getNY()
        nz = grid.getNZ()
        scaling_values = np.zeros((nx, ny, nz), dtype=np.float32)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    global_index = grid.global_index(ijk=(i, j, k))
                    scaling_values[i, j, k] = decay_obj(global_index)
        return scaling_values
