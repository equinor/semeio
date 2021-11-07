# pylint: disable=R0915
import yaml
import pytest
from res.enkf import EnKFMain, ResConfig
from ert_shared.libres_facade import LibresFacade

from semeio.workflows.localisation.local_config_script import LocalisationConfigJob
from semeio.workflows.localisation.localisation_config import (
    LocalisationConfig,
    get_max_gen_obs_size_for_expansion,
)
from semeio.workflows.localisation.local_script_lib import (
    get_obs_from_ert,
    get_param_from_ert,
    active_index_for_parameter,
    Parameters,
)

from xtgeo.surface.regular_surface import RegularSurface
import xtgeo
import numpy as np
import itertools


def verify_ministep_active_param(
    corr_spec_list, ert_local_config, ert_ensemble_config, ert_param_dict
):
    """
    Script to verify that the local config matches the specified user config for
    parameters of type GEN_KW and GEN_PARAM.
    Reports mismatch if found and silent if OK.
    Used for test purpose.
    """
    from res.enkf.enums.active_mode_enum import ActiveMode
    from res.enkf.enums.ert_impl_type_enum import ErtImplType

    updatestep = ert_local_config.getUpdatestep()
    for ministep in updatestep:
        # User specification
        corr_spec = get_corr_group_spec(corr_spec_list, ministep.name())
        param_dict = Parameters.from_list(corr_spec.param_group.result_items).to_dict()
        if len(param_dict) != ministep.numActiveData():
            raise ValueError(
                f"len(param_dict):{len(param_dict)}\n"
                f"active nodes i ministep:{ministep.numActiveData()} "
            )
        print(
            f"\nMinistep:{ministep.name()}\n"
            f"Number of parameter nodes:{len(param_dict)}"
        )

        for node_name, user_spec_param_list in param_dict.items():
            print(f"Node_name:{node_name} ")
            print(f"Param list:{user_spec_param_list} ")

            active_list_obj = ministep.getActiveList(node_name)
            node = ert_ensemble_config.getNode(node_name)
            impl_type = node.getImplementationType()

            # Check only cases with partly active set of parameter
            spec_index_list = []
            if active_list_obj.getMode() == ActiveMode.PARTLY_ACTIVE:
                if impl_type == ErtImplType.GEN_KW:
                    for user_param_name in user_spec_param_list:
                        spec_index_list.append(
                            active_index_for_parameter(
                                node_name, user_param_name, ert_param_dict
                            )
                        )

                elif impl_type == ErtImplType.GEN_DATA:
                    for item in user_spec_param_list:
                        spec_index_list.append(int(item))

                active_index_list = active_list_obj.get_active_index_list()
                spec_index_list.sort()
                active_index_list.sort()
                print(f"Spec index list:{spec_index_list} ")
                print(f"Ministep has active index list: {active_index_list} ")
                if len(spec_index_list) != len(active_index_list):
                    raise ValueError(
                        f"For ministep: {ministep.name()} the number of "
                        "active parameters are: "
                        f"{len(active_index_list)} \n"
                        "while the specified number of active parameters "
                        f"are: {len(spec_index_list)}"
                    )
                if active_index_list != spec_index_list:
                    raise ValueError(
                        f" For ministep: {ministep.name()} and "
                        f"parameter node: {node_name}:\n"
                        "Mismatch between specified active parameters "
                        f"and active parameters in the ministep.\n"
                        f"Specified: {spec_index_list}\n"
                        f"In ministep: {active_index_list}\n"
                    )


def verify_ministep_active_obs(corr_spec_list, ert):
    # pylint: disable=R1702
    """
    Script to verify that the local config matches the specified user config for
    active observations.
    Reports mismatch if found and silent if OK.
    Used for test purpose.
    """
    from res.enkf.enums.active_mode_enum import ActiveMode

    facade = LibresFacade(ert)
    ert_obs = facade.get_observations()
    ert_local_config = ert.getLocalConfig()

    updatestep = ert_local_config.getUpdatestep()
    for ministep in updatestep:
        # User specification
        corr_spec = get_corr_group_spec(corr_spec_list, ministep.name())
        obs_dict = Parameters.from_list(corr_spec.obs_group.result_items).to_dict()

        # Data from local config, only one obs group in a ministep here.
        local_obs_data = ministep.getLocalObsData()
        for obs_node in local_obs_data:
            key = obs_node.key()
            impl_type = facade.get_impl_type_name_for_obs_key(key)
            if impl_type == "GEN_OBS":
                active_list_obj = obs_node.getActiveList()
                if active_list_obj.getMode() == ActiveMode.PARTLY_ACTIVE:
                    obs_vector = ert_obs[key]
                    # Always 1 timestep for a GEN_OBS
                    timestep = obs_vector.activeStep()
                    genobs_node = obs_vector.getNode(timestep)
                    data_size = genobs_node.getSize()
                    active_list_obj = obs_node.getActiveList()
                    active_index_list = active_list_obj.get_active_index_list()
                    active_index_list.sort()
                    # From user specification
                    str_list = obs_dict[key]
                    spec_index_list = [int(str_list[i]) for i in range(len(str_list))]
                    spec_index_list.sort()
                    err = False
                    for nr, index in enumerate(active_index_list):
                        if index != spec_index_list[nr]:
                            err = True
                    if err:
                        raise ValueError(
                            f" For ministep: {ministep.name()} and "
                            f"observation node: {key}:\n"
                            "Mismatch between specified active observations and "
                            "active observations  defined in the ministep.\n"
                            f"Specified: {spec_index_list}\n"
                            f"In ministep: {active_index_list}\n"
                            f"Total number of observations for node {key} "
                            f"is {data_size}."
                        )


def get_corr_group_spec(correlations_spec_list, name):
    corr_spec_found = None
    for corr_spec in correlations_spec_list:
        if name == corr_spec.name:
            corr_spec_found = corr_spec
            break
    if not corr_spec_found:
        raise ValueError(
            f"Can not find correlation group: {name} in user specification."
        )
    return corr_spec_found


def check_consistency_for_active_param_and_obs(ert, config_dict):
    expand_gen_obs_max_size = get_max_gen_obs_size_for_expansion(config_dict)
    obs_keys = get_obs_from_ert(ert, expand_gen_obs_max_size)
    ert_parameters = get_param_from_ert(ert.ensembleConfig())
    config = LocalisationConfig(
        observations=obs_keys,
        parameters=ert_parameters.to_list(),
        **config_dict,
    )
    verify_ministep_active_param(
        config.correlations,
        ert.getLocalConfig(),
        ert.ensembleConfig(),
        ert_parameters.to_dict(),
    )
    verify_ministep_active_obs(config.correlations, ert)


@pytest.mark.parametrize(
    "obs_group_add, param_group_add, expected",
    [
        (
            ["FOPR", "WOPR_OP1_190"],
            ["SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE", "SNAKE_OIL_PARAM:OP1_OFFSET"],
            ["SNAKE_OIL_PARAM"],
        ),
    ],
)
def test_localisation(setup_ert, obs_group_add, param_group_add, expected):
    ert = EnKFMain(setup_ert)
    config_dict = {
        "log_level": 4,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {"add": "*", "remove": obs_group_add},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                },
            },
            {
                "name": "CORR2",
                "obs_group": {"add": "*", "remove": obs_group_add},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:*",
                    "remove": "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                },
            },
            {
                "name": "CORR3",
                "obs_group": {"add": obs_group_add},
                "param_group": {
                    "add": param_group_add,
                },
            },
        ],
    }
    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")
    assert ert.getLocalConfig().getMinistep("CORR1").name() == "CORR1"
    assert (
        ert.getLocalConfig().getObsdata("CORR1_obs_group").name() == "CORR1_obs_group"
    )
    result = {}
    for index, ministep in enumerate(ert.getLocalConfig().getUpdatestep()):
        result[ministep.name()] = {
            "obs": [obs_node.key() for obs_node in ministep.getLocalObsData()],
            "key": ministep.name() + "_param_group",
        }
    expected_result = {
        "CORR1": {
            "obs": [
                "WOPR_OP1_108",
                "WOPR_OP1_144",
                "WOPR_OP1_36",
                "WOPR_OP1_72",
                "WOPR_OP1_9",
                "WPR_DIFF_1",
            ],
            "key": "CORR1_param_group",
        },
        "CORR2": {
            "obs": [
                "WOPR_OP1_108",
                "WOPR_OP1_144",
                "WOPR_OP1_36",
                "WOPR_OP1_72",
                "WOPR_OP1_9",
                "WPR_DIFF_1",
            ],
            "key": "CORR2_param_group",
        },
        "CORR3": {"obs": ["FOPR", "WOPR_OP1_190"], "key": "CORR3_param_group"},
    }
    assert result == expected_result


def test_localisation_gen_kw(setup_ert):
    ert = EnKFMain(setup_ert, verbose=True)
    config_dict = {
        "log_level": 4,
        "max_gen_obs_size": 1000,
        "correlations": [
            {
                "name": "CORR12",
                "obs_group": {"add": ["WPR_DIFF_1:0", "WPR_DIFF_1:3"]},
                "param_group": {
                    "add": [
                        "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                        "SNAKE_OIL_PARAM:OP1_OCTAVES",
                    ],
                },
            },
            {
                "name": "CORR3",
                "obs_group": {"add": "WPR_DIFF_1:2"},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
                },
            },
            {
                "name": "CORR4",
                "obs_group": {
                    "add": "*",
                    "remove": ["WPR_DIFF_1:1", "WPR_DIFF_1:0"],
                },
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:OP1_OFFSET",
                },
            },
            {
                "name": "CORR5",
                "obs_group": {"add": "*"},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:OP2_PERSISTENCE",
                },
            },
            {
                "name": "CORR6",
                "obs_group": {"add": "*"},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:OP2_OCTAVES",
                },
            },
            {
                "name": "CORR789",
                "obs_group": {"add": "*"},
                "param_group": {
                    "add": [
                        "SNAKE_OIL_PARAM:OP2_DIVERGENCE_SCALE",
                        "SNAKE_OIL_PARAM:OP2_OFFSET",
                        "SNAKE_OIL_PARAM:BPR_555_PERSISTENCE",
                    ],
                },
            },
            {
                "name": "CORR10",
                "obs_group": {"add": "*"},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:BPR_138_PERSISTENCE",
                },
            },
        ],
    }
    with open("local_config_gen_kw.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config_gen_kw.yaml")
    check_consistency_for_active_param_and_obs(ert, config_dict)


def test_localisation_gen_param_and_gen_obs_expanded(
    setup_poly_gen_param_ert,
):
    """
    POLY_OBS is a GEN_OBS node with less than 10 observations
    In this case the GEN_OBS node is expanded and the user specify
    obs like POLY_OBS:index
    """
    ert = EnKFMain(setup_poly_gen_param_ert, verbose=True)
    fs = ert.getEnkfFsManager().getFileSystem("default_smoother_update")
    ert.getEnkfFsManager().switchFileSystem(fs)
    config_dict = {
        "log_level": 2,
        "max_gen_obs_size": 10,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {"add": ["POLY_OBS:0", "POLY_OBS:3", "POLY_OBS:4"]},
                "param_group": {"add": ["COEFFS:COEFF_B", "PARAMS_A"]},
            },
        ],
    }
    with open("local_config_gen_param.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config_gen_param.yaml")
    check_consistency_for_active_param_and_obs(ert, config_dict)


def test_localisation_gen_param_and_gen_obs_not_expanded1(
    setup_poly_gen_param_ert,
):
    """
    POLY_OBS is a GEN_OBS node with less than 10 observations
    In this case the GEN_OBS node is not expanded since
    max_gen_obs_size < number of obs in the GEN_OBS node.
    """

    ert = EnKFMain(setup_poly_gen_param_ert, verbose=True)
    fs = ert.getEnkfFsManager().getFileSystem("default_smoother_update")
    ert.getEnkfFsManager().switchFileSystem(fs)
    config_dict = {
        "log_level": 2,
        #        "max_gen_obs_size": 3,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {"add": ["POLY_OBS", "POLY_OBS", "POLY_OBS"]},
                "param_group": {"add": ["COEFFS:COEFF_B", "PARAMS_A"]},
            },
        ],
    }
    with open("local_config_gen_param.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config_gen_param.yaml")
    check_consistency_for_active_param_and_obs(ert, config_dict)


def test_localisation_surf(
    setup_poly_ert,
):
    with open("poly.ert", "a", encoding="utf-8") as fout:
        fout.write(
            "SURFACE   PARAM_SURF_A     OUTPUT_FILE:surf.txt    "
            "INIT_FILES:surf%d.txt   BASE_SURFACE:surf0.txt"
        )
    nreal = 20
    ncol = 10
    nrow = 10
    rotation = 0.0
    xinc = 50.0
    yinc = 50.0
    xori = 0.0
    yori = 0.0
    values = np.zeros(nrow * ncol)
    for n in range(nreal):
        filename = "surf" + str(n) + ".txt"
        delta = 0.1
        for j in range(nrow):
            for i in range(ncol):
                index = i + j * ncol
                values[index] = float(j) + n * delta
        surface = RegularSurface(
            ncol=ncol,
            nrow=nrow,
            xinc=xinc,
            yinc=yinc,
            xori=xori,
            yori=yori,
            rotation=rotation,
            values=values,
        )
        surface.to_file(filename, fformat="irap_ascii")

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config_dict = {
        "log_level": 3,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "surface_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "azimuth": 200,
                    "ref_point": [250, 250],
                    "surface_file": "surf0.txt",
                },
            },
        ],
    }

    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


# This test and the test test_localisation_field2 are similar,
# but the first test a case with multiple fields and multiple
# ministeps where write_scaling_factor is activated and one
# file is written per ministep.
# Test case 2 tests three different methods for defining scaling factors for fields
def test_localisation_field1(
    setup_poly_ert,
):
    # Make a 3D grid with no inactive cells
    grid_filename = "grid3D.EGRID"
    grid = create_box_grid_with_inactive_and_active_cells(
        grid_filename, has_inactive_values=False
    )

    nreal = 20
    (nx, ny, nz) = grid.dimensions
    with open("poly.ert", "a", encoding="utf-8") as fout:
        fout.write(f"GRID   {grid_filename}\n")

        property_names = ["G1", "G2", "G3", "G4"]
        for pname in property_names:
            filename_output = pname + ".roff"
            filename_input = pname + "_%d.roff"
            values = np.zeros((nx, ny, nz), dtype=np.float32)
            property_field = xtgeo.GridProperty(grid, values=0.0, name=pname)
            for n in range(nreal):
                values = np.zeros((nx, ny, nz), dtype=np.float32)
                property_field.values = values + 0.1 * n
                filename = pname + "_" + str(n) + ".roff"
                property_field.to_file(filename, fformat="roff", name=pname)

            fout.write(
                f"FIELD  {pname}  PARAMETER  {filename_output}  "
                f"INIT_FILES:{filename_input}  MIN:-5.5   MAX:5.5  "
                "FORWARD_INIT:False\n"
            )

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config_dict = {
        "log_level": 3,
        "write_scaling_factors": True,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G1", "G2"],
                },
                "field_scale": {
                    "method": "exponential_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "azimuth": 200,
                    "ref_point": [700, 370],
                },
            },
            {
                "name": "CORR2",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G3"],
                },
                "field_scale": {
                    "method": "const_gaussian_decay",
                    "main_range": 1000,
                    "perp_range": 950,
                    "azimuth": 100,
                    "ref_point": [700, 370],
                },
            },
            {
                "name": "CORR3",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": ["G4"],
                },
                "field_scale": {
                    "method": "const_exponential_decay",
                    "main_range": 1000,
                    "perp_range": 950,
                    "azimuth": 100,
                    "ref_point": [700, 370],
                    "normalised_tapering_range": 1.2,
                    "cutoff": True,
                },
            },
        ],
    }

    with open("local_config.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


def create_box_grid_with_inactive_and_active_cells(
    output_grid_file, has_inactive_values=True
):
    # pylint: disable=E1120
    nx = 30
    ny = 25
    nz = 3
    xinc = 50.0
    yinc = 50.0
    zinc = 10.0
    xori = 0.0
    yori = 0.0
    grid = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(xori, yori, 0.0),
        increment=(xinc, yinc, zinc),
        rotation=0.0,
        flip=1,
    )
    # Create a polygon file to use to set some grid cells inactive
    with open("polygon.txt", "w") as fout:
        x = []
        y = []
        x.append(xori + 5 * xinc)
        y.append(yori + 5 * yinc)

        x.append(xori + (nx - 6) * xinc)
        y.append(yori + 5 * yinc)

        x.append(xori + (nx - 6) * xinc)
        y.append(yori + (ny - 6) * yinc)

        x.append(xori + 5 * xinc)
        y.append(yori + (ny - 6) * yinc)

        x.append(xori + 5 * xinc)
        y.append(yori + 5 * yinc)

        for i in range(5):
            fout.write(f" {x[i]}  {y[i]}  {zinc}\n")

    polygon = xtgeo.xyz.Polygons()
    polygon.from_file("polygon.txt", fformat="xyz")
    if has_inactive_values:
        grid.inactivate_outside(polygon, force_close=True)

    grid.to_file(output_grid_file, fformat="egrid")
    return grid


def create_region_parameter(filename, grid):
    # Create a discrete parameter to represent a region parameter
    region_param_name = "Region"
    region_code_names = {
        "RegionA": 1,
        "RegionB": 2,
        "RegionC": 3,
        "RegionD": 4,
        "RegionE": 5,
        "RegionF": 6,
    }
    region_param = xtgeo.GridProperty(
        grid, name=region_param_name, discrete=True, values=1
    )
    region_param.dtype = np.uint16
    region_param.codes = region_code_names
    (nx, ny, nz) = grid.dimensions
    values = np.zeros((nx, ny, nz), dtype=np.uint16)
    values[:, :, :] = 0
    for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
        if 0 <= i <= nx / 2 and 0 <= j <= ny / 2:
            if 0 <= k <= nz / 2:
                values[i, j, k] = 2
            else:
                values[i, j, k] = 5
        if nx / 2 + 1 <= i < nx and 0 <= j <= ny / 2:
            if nz / 2 <= k < nz:
                values[i, j, k] = 3
            else:
                values[i, j, k] = 4
        if ny / 3 + 1 <= j < 2 * ny / 3 and nx / 3 <= i <= nx / 2:
            if nz / 4 <= k < nz / 2:
                values[i, j, k] = 6
            else:
                values[i, j, k] = 4
    region_param.values = values
    region_param.to_file(filename, fformat="grdecl", name=region_param_name)


def create_field_and_scaling_param_and_update_poly_ert(
    poly_config_file, grid_filename, grid
):
    (nx, ny, nz) = grid.dimensions
    property_names = ["FIELD1", "FIELD2", "FIELD3", "FIELD4", "FIELD5"]
    scaling_names = ["SCALING1", "SCALING2", "SCALING3", "SCALING4", "SCALING5"]
    nreal = 20
    nfields = len(property_names)
    with open("poly.ert", "a") as fout:
        fout.write(f"GRID  {grid_filename}\n")
        for m in range(nfields):
            property_name = property_names[m]
            scaling_name = scaling_names[m]
            filename_output = property_name + ".roff"
            filename_input = property_name + "_%d.roff"
            scaling_filename = scaling_name + ".GRDECL"
            values = np.zeros((nx, ny, nz), dtype=np.float32)
            property_field = xtgeo.GridProperty(grid, values=0.0, name=property_name)
            scaling_field = xtgeo.GridProperty(
                grid, values=0.5 + (m - 1) * 0.2, name=scaling_name
            )
            for n in range(nreal):
                values = np.zeros((nx, ny, nz), dtype=np.float32)
                property_field.values = values + 0.1 * n
                filename = property_name + "_" + str(n) + ".roff"
                property_field.to_file(filename, fformat="roff", name=property_name)

            scaling_field.to_file(scaling_filename, fformat="grdecl", name=scaling_name)

            fout.write(
                f"FIELD  {property_name}  PARAMETER  {filename_output}  "
                f"INIT_FILES:{filename_input}  "
                "MIN:-5.5   MAX:5.5     FORWARD_INIT:False\n"
            )


def test_localisation_field2(setup_poly_ert):
    # Make a 3D grid with some inactive cells
    grid_filename = "grid3D.EGRID"
    grid = create_box_grid_with_inactive_and_active_cells(grid_filename)

    # Make some field parameters and some scalingfactor parameters
    poly_config_file = "poly.ert"
    create_field_and_scaling_param_and_update_poly_ert(
        poly_config_file, grid_filename, grid
    )

    # Create a discrete parameter to represent a region parameter
    segment_filename1 = "Region1.GRDECL"
    create_region_parameter(segment_filename1, grid)
    segment_filename2 = "Region2.GRDECL"
    create_region_parameter(segment_filename2, grid)

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config_dict = {
        "log_level": 3,
        "write_scaling_factors": True,
        "correlations": [
            {
                "name": "CORR_GAUSSIAN",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD1",
                },
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 700,
                    "perp_range": 150,
                    "azimuth": 30,
                    "ref_point": [500, 0],
                },
            },
            {
                "name": "CORR_FROM_FILE",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD2",
                },
                "field_scale": {
                    "method": "from_file",
                    "filename": "SCALING2.GRDECL",
                    "param_name": "SCALING2",
                },
            },
            {
                "name": "CORR_SEGMENT1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD3",
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": segment_filename1,
                    "param_name": "Region",
                    "active_segments": [1, 2, 4, 5],
                    "scalingfactors": [1.0, 1.5e-5, 0.3, 0.15],
                    "smooth_ranges": [2, 2],
                },
            },
            {
                "name": "CORR_SEGMENT2",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD4",
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": segment_filename1,
                    "param_name": "Region",
                    "active_segments": [1, 2, 4, 5],
                    "scalingfactors": [0.5, 1.0, 0.8, 0.05],
                    "smooth_ranges": [2, 2],
                },
            },
            {
                "name": "CORR_SEGMENT3",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "FIELD5",
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": segment_filename2,
                    "param_name": "Region",
                    "active_segments": [1, 3, 5],
                    "scalingfactors": [1.0, 0.5, 0.05],
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


def test_localisation_gen_obs(
    setup_poly_ert,
):
    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config_dict = {
        "log_level": 2,
        "max_gen_obs_size": 1000,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": ["POLY_OBS:*"],
                },
                "param_group": {
                    "add": ["*"],
                },
            },
        ],
    }
    with open("local_config_gen_obs.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config_gen_obs.yaml")
    check_consistency_for_active_param_and_obs(ert, config_dict)


@pytest.mark.parametrize(
    "obs_group_add1, obs_group_remove1, obs_group_add2, obs_group_remove2, expected",
    [
        (
            ["POLY_OBS:0", "POLY_OBS:1", "POLY_OBS:2"],
            [],
            ["POLY_OBS:3", "POLY_OBS:4"],
            ["POLY_OBS:3"],
            {
                "CORR1": [0, 1, 2],
                "CORR2": [4],
            },
        ),
        (
            ["POLY_OBS:*"],
            ["POLY_OBS:1*", "POLY_OBS:3"],
            ["POLY_OBS:3"],
            ["POLY_OBS:1"],
            {
                "CORR1": [0, 2, 4],
                "CORR2": [3],
            },
        ),
    ],
)
def test_localisation_gen_obs2(
    setup_poly_ert,
    obs_group_add1,
    obs_group_remove1,
    obs_group_add2,
    obs_group_remove2,
    expected,
):
    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config_dict = {
        "log_level": 2,
        "max_gen_obs_size": 1000,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": obs_group_add1,
                    "remove": obs_group_remove1,
                },
                "param_group": {
                    "add": ["*"],
                },
            },
            {
                "name": "CORR2",
                "obs_group": {
                    "add": obs_group_add2,
                    "remove": obs_group_remove2,
                },
                "param_group": {
                    "add": ["*"],
                },
            },
        ],
    }
    with open("local_config_gen_obs2.yaml", "w", encoding="utf-8") as fout:
        yaml.dump(config_dict, fout)
    LocalisationConfigJob(ert).run("local_config_gen_obs2.yaml")
    check_consistency_for_active_param_and_obs(ert, config_dict)
