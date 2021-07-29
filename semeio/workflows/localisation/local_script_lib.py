# pylint: disable=W0201
import yaml
import math
import cwrap
import numpy as np

from ecl.util.geometry import Surface
from res.enkf.enums.ert_impl_type_enum import ErtImplType
from res.enkf.enums.enkf_var_type_enum import EnkfVarType
from dataclasses import dataclass

from semeio.workflows.localisation.localisation_debug_settings import (
    LogLevel,
    debug_print,
)
import semeio.workflows.localisation.localisation_debug_settings as log_level_setting

# The global variables are only used for controlling output of log info to screen
# and for writing scaling factor to file (for QC and test purpose)
debug_level = log_level_setting.debug_level
scaling_parameter_number = log_level_setting.scaling_parameter_number


def get_observations_from_ert(ert):
    # Get list of observation nodes from ERT config to be used in consistency check
    ert_obs = ert.getObservations()
    obs_data_from_ert_config = ert_obs.getAllActiveLocalObsdata()
    ert_obs_node_list = []
    for obs_node in obs_data_from_ert_config:
        node_name = obs_node.key()
        ert_obs_node_list.append(node_name)
    return ert_obs_node_list


def get_param_from_ert(ert):
    ens_config = ert.ensembleConfig()
    keylist = ens_config.alloc_keylist()
    parameters_for_node = {}
    node_type = {}
    implementation_type_not_scalar = [
        ErtImplType.GEN_DATA,
        ErtImplType.FIELD,
        ErtImplType.SURFACE,
    ]
    for key in keylist:
        node = ens_config.getNode(key)
        impl_type = node.getImplementationType()
        node_type[key] = impl_type
        var_type = node.getVariableType()
        if var_type == EnkfVarType.PARAMETER:
            if impl_type == ErtImplType.GEN_KW:
                # Node contains scalar parameters defined by GEN_KW
                kw_config_model = node.getKeywordModelConfig()
                string_list = kw_config_model.getKeyWords()
                param_list = []
                for name in string_list:
                    param_list.append(name)
                parameters_for_node[key] = param_list
            elif impl_type in implementation_type_not_scalar:
                # Node contains parameter from FIELD, SURFACE or GEN_PARAM
                # The parameters_for_node dict contains empty list for FIELD
                # and SURFACE.
                # The number of variables for a FIELD parameter is
                # defined by the grid in the GRID keyword.
                # The number of variables for a SURFACE parameter is
                # defined by the size of a surface object.
                # For GEN_PARAM the list contains the number of variables.
                parameters_for_node[key] = []
                if impl_type == ErtImplType.GEN_DATA:
                    gen_data_config = node.getDataModelConfig()
                    data_size = gen_data_config.get_initial_size()
                    if data_size <= 0:
                        print(
                            f"\n -- Warning: A GEN_PARAM node {key} "
                            "is defined in ERT config.\n"
                            "          This requires that the localization script "
                            "must be run AFTER \n"
                            "          initialization of the ensemble, "
                            "but BEFORE the first update step.\n"
                        )
                    debug_print(
                        f"GEN_PARAM node {key} has {data_size} parameters.",
                        LogLevel.LEVEL3,
                    )
                    parameters_for_node[key] = [str(data_size)]

    return parameters_for_node, node_type


def read_localisation_config(args):
    if len(args) == 1:
        specification_file_name = args[0]
    else:
        raise ValueError(f"Expecting a single argument. Got {args} arguments.")

    print(
        "\n" f"Define localisation setup using config file: {specification_file_name}"
    )
    with open(specification_file_name, "r") as yml_file:
        localisation_yml = yaml.safe_load(yml_file)
    return localisation_yml


def active_index_for_parameter(node_name, param_name, ert_param_dict):
    # For parameters defined as scalar parameters (coming from GEN_KW)
    # the parameters for a node have a name. Get the index from the order
    # of these parameters.
    # For parameters that are not coming from GEN_KW,
    # the active index is not used here.

    ert_param_list = ert_param_dict[node_name]
    if len(ert_param_list) > 0:
        index = -1
        for count, name in enumerate(ert_param_list):
            if name == param_name:
                index = count
                break
        assert index > -1
    else:
        index = None
    return index


def activate_gen_kw_param(model_param_group, node_name, param_list, ert_param_dict):
    """
    Activate the selected parameters for the specified node.
    The param_list contains the list of parameters defined in GEN_KW
    for this node to be activated.
    """
    active_param_list = model_param_group.getActiveList(node_name)
    debug_print("Set active parameters", LogLevel.LEVEL2)
    for param_name in param_list:
        index = active_index_for_parameter(node_name, param_name, ert_param_dict)
        if index is not None:
            debug_print(
                f"Active parameter: {param_name}  index: {index}", LogLevel.LEVEL3
            )
            active_param_list.addActiveIndex(index)


def activate_gen_param(model_param_group, node_name, param_list, data_size):
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
        debug_print(f"Active parameter index: {index}", LogLevel.LEVEL3)
        active_param_list.addActiveIndex(index)


def write_scaling_values(
    corr_name,
    node_name,
    method_name,
    grid_for_field,
    ref_pos,
    main_range,
    perp_range,
    azimuth,
):
    # pylint: disable=W0603
    global scaling_parameter_number
    nx = grid_for_field.getNX()
    ny = grid_for_field.getNY()
    nz = grid_for_field.getNZ()
    if method_name == "gaussian_decay":
        decay_obj = GaussianDecay(
            ref_pos, main_range, perp_range, azimuth, grid_for_field, None
        )
    elif method_name == "exponential_decay":
        decay_obj = ExponentialDecay(
            ref_pos, main_range, perp_range, azimuth, grid_for_field, None
        )
    else:
        raise KeyError(f" Method name: {method_name} is not implemented")

    scaling_values = np.zeros((nx, ny, nz), dtype=np.float32)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                global_index = grid_for_field.global_index(ijk=(i, j, k))
                scaling_values[i, j, k] = decay_obj(global_index)

    scaling_kw_name = "S_" + str(scaling_parameter_number)
    scaling_kw = grid_for_field.create_kw(scaling_values, scaling_kw_name, False)
    filename = corr_name + "_" + node_name + "_scaling" + ".GRDECL"
    print(f"   -- Write file: {filename}")
    with cwrap.open(filename, "w") as file:
        grid_for_field.write_grdecl(scaling_kw, file)
    scaling_parameter_number += 1


def activate_and_scale_field_correlation(
    model_param_group, node_name, field_config, corr_spec, grid_for_field
):
    assert corr_spec.field_scale
    ref_pos = corr_spec.ref_point
    check_if_ref_point_in_grid(ref_pos, grid_for_field)

    # A scaling of correlation is defined
    debug_print(
        f"Apply the correlation scaling method: {corr_spec.field_scale.method}",
        LogLevel.LEVEL3,
    )

    row_scaling = model_param_group.row_scaling(node_name)
    data_size = field_config.get_data_size()

    if corr_spec.field_scale.method == "gaussian_decay":
        main_range = corr_spec.field_scale.main_range
        perp_range = corr_spec.field_scale.perp_range
        azimuth = corr_spec.field_scale.angle
        row_scaling.assign(
            data_size,
            GaussianDecay(
                ref_pos, main_range, perp_range, azimuth, grid_for_field, None
            ),
        )
        if debug_level >= LogLevel.LEVEL4:
            write_scaling_values(
                corr_spec.name,
                node_name,
                corr_spec.field_scale.method,
                grid_for_field,
                ref_pos,
                main_range,
                perp_range,
                azimuth,
            )

    elif corr_spec.field_scale.method == "exponential_decay":
        main_range = corr_spec.field_scale.main_range
        perp_range = corr_spec.field_scale.perp_range
        azimuth = corr_spec.field_scale.angle
        row_scaling.assign(
            data_size,
            ExponentialDecay(
                ref_pos, main_range, perp_range, azimuth, grid_for_field, None
            ),
        )
        if debug_level >= LogLevel.LEVEL4:
            write_scaling_values(
                corr_spec.name,
                node_name,
                corr_spec.field_scale.method,
                grid_for_field,
                ref_pos,
                main_range,
                perp_range,
                azimuth,
            )

    else:
        print(
            f" --  Scaling method: {corr_spec.field_scale.method} "
            "is not implemented."
        )


def activate_and_scale_surface_correlation(
    model_param_group,
    node_name,
    corr_spec,
):
    assert corr_spec.surface_scale
    # A scaling of correlation is defined
    debug_print(
        "Get surface grid attributes from file: " f"{corr_spec.surface_scale.filename}",
        LogLevel.LEVEL2,
    )
    surface = Surface(corr_spec.surface_scale.filename)
    nx = surface.getNX()
    ny = surface.getNY()
    data_size = nx * ny
    debug_print(f"Surface grid size: ({nx}, {ny})", LogLevel.LEVEL3)
    debug_print(
        "Apply the correlation scaling method: " f"{corr_spec.surface_scale.method}\n",
        LogLevel.LEVEL2,
    )
    row_scaling = model_param_group.row_scaling(node_name)

    if corr_spec.surface_scale.method == "gaussian_decay":
        assert corr_spec.ref_point
        ref_pos = corr_spec.ref_point
        main_range = corr_spec.surface_scale.main_range
        perp_range = corr_spec.surface_scale.perp_range
        azimuth = corr_spec.surface_scale.angle
        row_scaling.assign(
            data_size,
            GaussianDecay(ref_pos, main_range, perp_range, azimuth, None, surface),
        )
    elif corr_spec.surface_scale.method == "exponential_decay":
        assert corr_spec.ref_point
        ref_pos = corr_spec.ref_point
        main_range = corr_spec.surface_scale.main_range
        perp_range = corr_spec.surface_scale.perp_range
        azimuth = corr_spec.surface_scale.angle
        row_scaling.assign(
            data_size,
            ExponentialDecay(ref_pos, main_range, perp_range, azimuth, None, surface),
        )
    else:
        print(
            f" --  Scaling method: {corr_spec.field_scale.method} "
            "is not implemented."
        )


def add_ministeps(user_config, ert_param_dict, ert):
    # pylint: disable-msg=too-many-branches
    local_config = ert.getLocalConfig()
    updatestep = local_config.getUpdatestep()
    ens_config = ert.ensembleConfig()
    grid_for_field = ert.eclConfig().getGrid()
    debug_print(f"Log level: {log_level_setting.debug_level}", LogLevel.LEVEL1)
    debug_print("Add all ministeps:", LogLevel.LEVEL1)
    if grid_for_field is not None:
        debug_print(f"Get grid: {grid_for_field.get_name()}", LogLevel.LEVEL1)

    for count, corr_spec in enumerate(user_config.correlations):
        ministep_name = corr_spec.name
        ministep = local_config.createMinistep(ministep_name)
        debug_print(f"Define ministep: {ministep_name}", LogLevel.LEVEL1)

        param_group_name = ministep_name + "_param_group"
        model_param_group = local_config.createDataset(param_group_name)

        obs_group_name = ministep_name + "_obs_group"
        obs_group = local_config.createObsdata(obs_group_name)

        obs_list = corr_spec.obs_group.add
        param_dict = corr_spec.param_group.add

        # Setup model parameter group
        for node_name, param_list in param_dict.items():
            node = ens_config.getNode(node_name)
            impl_type = node.getImplementationType()

            debug_print(f"Add node: {node_name} of type: {impl_type}", LogLevel.LEVEL2)
            model_param_group.addNode(node_name)
            if impl_type == ErtImplType.GEN_KW:
                activate_gen_kw_param(
                    model_param_group, node_name, param_list, ert_param_dict
                )
            elif impl_type == ErtImplType.FIELD:
                if corr_spec.field_scale is not None:
                    field_config = node.getFieldModelConfig()

                    activate_and_scale_field_correlation(
                        model_param_group,
                        node_name,
                        field_config,
                        corr_spec,
                        grid_for_field,
                    )
                else:
                    debug_print(
                        f"No correlation scaling for node {node_name} "
                        f"in {ministep_name}",
                        LogLevel.LEVEL3,
                    )
            elif impl_type == ErtImplType.GEN_DATA:
                gen_data_config = node.getDataModelConfig()
                data_size = gen_data_config.get_initial_size()
                if data_size > 0:
                    activate_gen_param(
                        model_param_group, node_name, param_list, data_size
                    )
                else:
                    debug_print(
                        f"Parameter {node_name} has data size: {data_size} "
                        f"in {ministep_name}",
                        LogLevel.LEVEL3,
                    )
            elif impl_type == ErtImplType.SURFACE:
                if corr_spec.surface_scale is not None:
                    activate_and_scale_surface_correlation(
                        model_param_group,
                        node_name,
                        corr_spec,
                    )
                else:
                    debug_print(
                        f"No correlation scaling for node {node_name} "
                        f"in {ministep_name}",
                        LogLevel.LEVEL3,
                    )

        # Setup observation group
        for obs_name in obs_list:
            debug_print(f"Add obs node: {obs_name}", LogLevel.LEVEL2)
            obs_group.addNode(obs_name)

        # Setup ministep
        debug_print(
            f"Attach {param_group_name} to ministep {ministep_name}", LogLevel.LEVEL1
        )
        ministep.attachDataset(model_param_group)

        debug_print(
            f"Attach {obs_group_name} to ministep {ministep_name}", LogLevel.LEVEL1
        )
        ministep.attachObsset(obs_group)

        debug_print(f"Add {ministep_name} to update step\n", LogLevel.LEVEL1)
        updatestep.attachMinistep(ministep)


def clear_correlations(ert):
    local_config = ert.getLocalConfig()
    local_config.clear()


def check_if_ref_point_in_grid(ref_point, grid):
    try:
        i, j = grid.find_cell_xy(ref_point[0], ref_point[1], 0)
    except ValueError:
        raise ValueError(
            f"Reference point {ref_point} corresponds to undefined grid cell "
            f"or is outside the area defined by the grid {grid.get_name()}\n"
            "Check specification of reference point."
        ) from None


# This is an example callable which decays as a gaussian away from a position
@dataclass
class Decay:
    obs_pos: list
    main_range: float
    perp_range: float
    azimuth: float
    grid3D: object
    grid2D: object

    def __post_init__(self):
        angle = (90.0 - self.azimuth) * math.pi / 180.0
        self.cosangle = math.cos(angle)
        self.sinangle = math.sin(angle)

    def get_dx_dy(self, data_index):
        if self.grid3D is not None:
            x, y, _ = self.grid3D.get_xyz(active_index=data_index)
        elif self.grid2D is not None:
            x, y = self.grid2D.getXY(data_index)
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
