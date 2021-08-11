import yaml
import pytest
from res.enkf import EnKFMain, ResConfig
from semeio.workflows.localisation.local_config_script import LocalisationConfigJob

from xtgeo.surface.regular_surface import RegularSurface
import xtgeo
import numpy as np


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
    config = {
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
    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
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


def test_localisation_gen_param(
    setup_poly_ert,
):
    with open("poly.ert", "a") as fout:
        fout.write(
            "GEN_PARAM PARAMS_A parameter_file_A INPUT_FORMAT:ASCII "
            "OUTPUT_FORMAT:ASCII INIT_FILES:initial_param_file_A_%d"
        )
    nreal = 5
    nparam = 10
    for n in range(nreal):
        filename = "initial_param_file_A_" + str(n)
        with open(filename, "w") as fout:
            for i in range(nparam):
                fout.write(f"{i}\n")

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


def test_localisation_surf(
    setup_poly_ert,
):
    with open("poly.ert", "a") as fout:
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
    config = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "ref_point": [250, 250],
                "surface_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "azimuth": 200,
                    "surf_filename": "surf0.txt",
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


def test_localisation_field(
    setup_poly_ert,
):
    nreal = 20
    nx = 20
    ny = 10
    nz = 3
    xinc = 50.0
    yinc = 50.0
    zinc = 10.0
    xori = 0.0
    yori = 0.0
    grid = xtgeo.Grid()
    grid.create_box(
        dimension=(nx, ny, nz),
        origin=(xori, yori, 0.0),
        increment=(xinc, yinc, zinc),
        rotation=30.0,
        flip=-1,
    )
    grid_file_name = "grid3D.EGRID"
    print(f" Write file: {grid_file_name}")
    grid.to_file(grid_file_name, fformat="egrid")
    with open("poly.ert", "a") as fout:
        fout.write(f"GRID   {grid_file_name}\n")

        property_names = ["G1", "G2"]
        for pname in property_names:
            filename_output = pname + ".roff"
            filename_input = pname + "_%d.roff"
            values = np.zeros((nx, ny, nz), dtype=np.float32)
            property_field = xtgeo.GridProperty(grid, values=0.0, name=pname)
            for n in range(nreal):
                values = np.zeros((nx, ny, nz), dtype=np.float32)
                property_field.values = values + 0.1 * n
                filename = pname + "_" + str(n) + ".roff"
                print(f"Write file: {filename}")
                property_field.to_file(filename, fformat="roff", name=pname)

            fout.write(
                f"FIELD  {pname}  PARAMETER  {filename_output}  "
                f"INIT_FILES:{filename_input}  MIN:-5.5   MAX:5.5  "
                "FORWARD_INIT:False\n"
            )

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config = {
        "log_level": 4,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "ref_point": [700, 370],
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "azimuth": 200,
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")
