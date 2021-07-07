import yaml
import pytest
from res.enkf import EnKFMain, ResConfig
from semeio.workflows.localisation.local_config_script import LocalisationConfigJob


@pytest.mark.parametrize(
    "obs_group_add, param_group_add, expected, expected_obs1, expected_obs2",
    [
        (
            ["FOPR", "WOPR_OP1_190"],
            ["SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE", "SNAKE_OIL_PARAM:OP1_OFFSET"],
            ["SNAKE_OIL_PARAM"],
            [
                "WOPR_OP1_108",
                "WOPR_OP1_144",
                "WOPR_OP1_36",
                "WOPR_OP1_72",
                "WOPR_OP1_9",
                "WPR_DIFF_1",
            ],
            ["FOPR", "WOPR_OP1_190"],
        ),
    ],
)
def test_localisation(
    setup_ert, obs_group_add, param_group_add, expected, expected_obs1, expected_obs2
):
    ert = EnKFMain(setup_ert)
    config = {
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
    assert len(ert.getLocalConfig().getUpdatestep()) == 3
    ministep_names = ["CORR1", "CORR2", "CORR3"]
    for index, ministep in enumerate(ert.getLocalConfig().getUpdatestep()):
        assert ministep.name() == ministep_names[index]
        obs_list = []
        for count, obsnode in enumerate(ministep.getLocalObsData()):
            obs_list.append(obsnode.key())
        obs_list.sort()

        if index in [0, 1]:
            assert obs_list == expected_obs1
        else:
            assert obs_list == expected_obs2
        key = ministep_names[index] + "_param_group"
        assert ministep[key].keys() == expected


def test_localisation_gen_param(
    setup_poly_ert,
):
    with open("poly.ert", "a") as fout:
        fout.write(
            "GEN_PARAM PARAMS_A parameter_file_A INPUT_FORMAT:ASCII "
            "OUTPUT_FORMAT:ASCII INIT_FILES:initial_param_file_A_%d"
        )
    with open("initial_param_file_A_0", "w") as fout:
        for i in range(10):
            fout.write(f"{i}\n")

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config = {
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
