from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from ert.storage import open_storage
from scipy import stats

from semeio._exceptions.exceptions import ValidationError
from semeio.workflows.ahm_analysis import ahmanalysis
from semeio.workflows.ahm_analysis.ahmanalysis import _run_ministep


def test_make_update_log_df(snake_oil_facade, snapshot):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    with open_storage(snake_oil_facade.enspath, "w") as storage:
        prior_ens = storage.get_ensemble_by_name("default")
        posterior_ens = storage.create_ensemble(
            prior_ens.experiment_id,
            ensemble_size=prior_ens.ensemble_size,
            iteration=1,
            name="new_ensemble",
            prior_ensemble=prior_ens,
        )
        log = _run_ministep(
            snake_oil_facade,
            prior_ens,
            posterior_ens,
            sorted(prior_ens.experiment.observations.keys()),
            sorted(prior_ens.experiment.parameter_configuration.keys()),
        )
    snapshot.assert_match(
        ahmanalysis.make_update_log_df(log).round(4).to_csv(),
        "update_log.csv",
    )


def test_count_active_observations():
    """test function creates a dataframe reporting active observations during update"""
    df_update_log = pd.DataFrame(
        columns=["obs_key", "status"],
        data=[
            ("RWI_3_OBS", "Active"),
            ("RWI_3_OBS", "Inactive"),
            ("RWI_2_OBS", "Active"),
            ("RWI_2_OBS", "Missing"),
            ("OP_1_WGOR2", "Active"),
        ],
    )
    result = ahmanalysis.count_active_observations(df_update_log)
    assert result == 3


@pytest.mark.parametrize(
    "input_obs, expected_misfit, update_log",
    [
        (
            "RWI_3_OBS",
            (2.0 + 10.0 / 3.0) / 2.0,
            {
                "obs_key": {0: "RWI_3_OBS", 1: "RWI_3_OBS", 2: "RWI_3_OBS"},
                "status": {0: "Active", 1: "Inactive", 2: "Active"},
            },
        ),
        (
            "All_obs",
            (7.0 / 3.0 + 11.5 / 3.0) / 2.0,
            {
                "obs_key": {
                    0: "RWI_3_OBS",
                    1: "RWI_3_OBS",
                    2: "RWI_3_OBS",
                    3: "OP_3_WWCT1",
                    4: "OP_3_WWCT2",
                },
                "status": {
                    0: "Active",
                    1: "Inactive",
                    2: "Missing",
                    3: "Active",
                    4: "Missing",
                },
            },
        ),
        (
            "OP_3_WWCT",
            (1.4 / 3.0 + 1.8 / 3.0) / 2.0,
            {
                "obs_key": {0: "OP_3_WWCT1", 1: "OP_3_WWCT2", 2: "OP_3_WWCT3"},
                "status": {0: "Active", 1: "Inactive", 2: "Active"},
            },
        ),
    ],
)
def test_calc_observationsgroup_misfit(input_obs, expected_misfit, update_log):
    """test function creates a dataframe
    reporting misfit data for each obs vector"""
    misfit_df = pd.DataFrame(
        {
            "MISFIT:RWI_3_OBS": [6, 10],
            "MISFIT:OP_3_WWCT1": [1, 1.5],
            "MISFIT:OP_3_WWCT2": [0.15, 0.2],
            "MISFIT:OP_3_WWCT3": [0.25, 0.1],
        }
    )
    update_log_df = pd.DataFrame(update_log)
    misfitval = ahmanalysis.calc_observationsgroup_misfit(
        input_obs, update_log_df, misfit_df
    )
    assert misfitval == expected_misfit


@pytest.mark.parametrize(
    "input_obs, expected_msg, update_log",
    [
        (
            "OP_1_WWCT",
            "WARNING: no MISFIT value for observation OP_1_WWCT",
            {"status": {0: "Mising"}},
        ),
    ],
)
def test_warning_calc_observationsgroup_misfit(input_obs, expected_msg, update_log):
    """test function creates a dataframe
    reporting misfit data for each obs vector except the ones with All_obs-"""
    misfit_df = pd.DataFrame(
        {
            "MISFIT:RWI_3_OBS": [6],
            "MISFIT:OP_3_WWCT1": [1],
            "MISFIT:OP_3_WWCT2": [0.25],
        }
    )
    update_log_df = pd.DataFrame(update_log)
    with pytest.warns(UserWarning, match=expected_msg):
        ahmanalysis.calc_observationsgroup_misfit(input_obs, update_log_df, misfit_df)


def test_calc_kolmogorov_smirnov():
    """test function creates a dataframe reporting
    ks value between 0 and 1 for a prior and posterior distribution"""
    ks_matrix = pd.DataFrame()
    dkeys = ["param1", "param2"]
    ks_matrix = pd.DataFrame(sorted(dkeys), columns=["Parameters"])
    rng = np.random.default_rng(12345678)
    prior_data = pd.DataFrame(
        {
            "param1": stats.norm.rvs(size=100, loc=0, scale=1, random_state=rng),
            "param2": stats.norm.rvs(size=100, loc=0.01, scale=1, random_state=rng),
        }
    )
    target_data = pd.DataFrame(
        {
            "param1": stats.norm.rvs(size=200, loc=0.5, scale=1.5, random_state=rng),
            "param2": stats.norm.rvs(size=200, loc=0.1, scale=1, random_state=rng),
        }
    )
    ks_matrix["WOPT:W1"] = ks_matrix["Parameters"].map(
        ahmanalysis.calc_kolmogorov_smirnov(dkeys, prior_data, target_data)
    )
    ks_matrix.set_index("Parameters", inplace=True)
    assert "param2" in ks_matrix.index
    assert ks_matrix.loc["param1", "WOPT:W1"] == 0.275
    assert ks_matrix["WOPT:W1"].max() <= 1
    assert ks_matrix["WOPT:W1"].min() >= 0


def test_check_names():
    """test function check names to be used for prior and update case"""
    ert_currentname = "default"
    prior_name = None
    target_name = "<ANALYSIS_CASE_NAME>"
    p_name, tname = ahmanalysis.check_names(ert_currentname, prior_name, target_name)
    assert p_name == ert_currentname
    assert tname != "<ANALYSIS_CASE_NAME>"


@pytest.mark.parametrize(
    "misfit_data, prior_data, expected_msg",
    [
        (
            pd.DataFrame(),
            pd.DataFrame({"KEY:OP1": [0, 1, 2]}),
            "Empty prior ensemble",
        ),
        (
            pd.DataFrame(
                {
                    "MISFIT:KEY": [6],
                }
            ),
            pd.DataFrame(),
            "Empty parameters set for History Matching",
        ),
    ],
)
def test_raise_if_empty(misfit_data, prior_data, expected_msg):
    """test function check that run fails if empty misfit or prior data"""
    with pytest.raises(ValidationError, match=expected_msg):
        ahmanalysis.raise_if_empty(
            dataframes=[prior_data, misfit_data],
            messages=[
                "Empty prior ensemble",
                "Empty parameters set for History Matching",
            ],
        )


@pytest.mark.parametrize(
    "input_map, expected_keys",
    [
        (
            {"data_key_1": ["obs_1", "obs_2", "obs_3"]},
            {"data_key_1": ["obs_1", "obs_2", "obs_3"]},
        ),
        (
            {"data_key_1": ["obs_1", "obs_2", "obs_3"], "data_key_2": ["obs_4"]},
            {
                "data_key_1": ["obs_1", "obs_2", "obs_3"],
                "data_key_2": ["obs_4"],
                "All_obs": ["obs_1", "obs_2", "obs_3", "obs_4"],
            },
        ),
        (
            {
                "data_key_1": ["obs_1", "obs_2", "obs_3"],
                "data_key_2": ["obs_4"],
                "data_key_3": ["obs_6"],
            },
            {
                "data_key_1": ["obs_1", "obs_2", "obs_3"],
                "data_key_2": ["obs_4"],
                "data_key_3": ["obs_6"],
                "All_obs-data_key_3": ["obs_1", "obs_2", "obs_3", "obs_4"],
                "All_obs-data_key_2": ["obs_1", "obs_2", "obs_3", "obs_6"],
                "All_obs-data_key_1": ["obs_4", "obs_6"],
                "All_obs": ["obs_1", "obs_2", "obs_3", "obs_4", "obs_6"],
            },
        ),
    ],
)
def test_make_obs_groups(input_map, expected_keys):
    result = ahmanalysis.make_obs_groups(input_map)
    assert result == expected_keys


@pytest.mark.parametrize(
    "prior_data, expected_result",
    [
        [
            {
                "SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
                "SNAKE_OIL_PARAM:OP1_OCTAVES": [0, 1, 2],
                "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": [0, 1, 2],
                "SNAKE_OIL_PARAM:OP1_OFFSET": [0, 0, 0],
                "SNAKE_OIL_PRES:BPR_138_PERSISTENCE": [0, 1, 2],
            },
            [
                "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                "SNAKE_OIL_PARAM:OP1_OCTAVES",
                "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
                "SNAKE_OIL_PRES:BPR_138_PERSISTENCE",
            ],
        ],
        [
            {
                "SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
                "SNAKE_OIL_PARAM:OP1_OCTAVES": [0, 1, 2],
                "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": [0, 1, 2],
                "SNAKE_OIL_PRES:BPR_138_PERSISTENCE": [0, 1, 2],
                "LOG10_SNAKE_OIL_PARAM:OP1_PERSISTENCE": [0, 1, 2],
            },
            [
                "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE",
                "SNAKE_OIL_PARAM:OP1_OCTAVES",
                "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                "SNAKE_OIL_PRES:BPR_138_PERSISTENCE",
            ],
        ],
    ],
)
def test_get_updated_parameters(prior_data, expected_result):
    """test function creates a dataframe with all scalar parameters"""
    prior_data = pd.DataFrame(prior_data)
    scalar_parameters = ["SNAKE_OIL_PARAM", "SNAKE_OIL_PRES"]
    p_keysf = ahmanalysis.get_updated_parameters(prior_data, scalar_parameters)
    assert sorted(p_keysf) == sorted(expected_result)


@pytest.mark.parametrize(
    "prior_data_w",
    [
        {
            "SNAKE_OIL_PARAM:OP1_PERSISTENCE": np.array([0, 1, 2]),
            "SNAKE_OIL_PARAM:OP1_OCTAVES": np.array([0, 1, 2]),
            "SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE": np.array([0, 1, 2]),
            "SNAKE_OIL_PARAM:OP1_OFFSET": np.array([[0, 0, 0], [0, 0, 0]]),
            "SNAKE_OIL_PRES:BPR_138_PERSISTENCE": np.array([0, 1, 2]),
        }
    ],
)
def test_warning_get_updated_parameters(prior_data_w):
    """test function creates a dataframe with all scalar parameters"""
    expected_msg = (
        "WARNING: Parameter SNAKE_OIL_PARAM:OP1_OFFSET defined several times."
    )
    scalar_parameters = ["SNAKE_OIL_PARAM", "SNAKE_OIL_PRES"]
    with pytest.warns(UserWarning, match=expected_msg):
        ahmanalysis.get_updated_parameters(prior_data_w, scalar_parameters)


def create_facade(keys):
    def side_effect(key):
        return keys[key]

    facade = MagicMock()
    facade.get_data_key_for_obs_key.side_effect = side_effect
    return facade


@pytest.mark.parametrize(
    "keys, expected_result",
    [
        pytest.param(
            {"obs_key": "data_key"},
            {"data_key": ["obs_key"]},
            id="One data key and one obs key",
        ),
        pytest.param(
            {"obs_key_1": "data_key", "obs_key_2": "data_key"},
            {"data_key": ["obs_key_1", "obs_key_2"]},
            id="Multiple obs per data key",
        ),
        pytest.param(
            {"obs:key_1": "data_key_1", "obs_key_2": "data_key_2"},
            {"data_key_1": ["obs_key_1"], "data_key_2": ["obs_key_2"]},
            id="Single obs for multiple data key",
        ),
        pytest.param(
            {
                "obs_key_1": "data_key_1",
                "obs_key_2": "data_key_2",
                "obs_key_3": "data_key_2",
            },
            {"data_key_1": ["obs_key_1"], "data_key_2": ["obs_key_2", "obs_key_3"]},
            id="Multiple obs for multiple data keys",
        ),
    ],
)
def test_group_obs_data_key(keys, expected_result):
    facade = create_facade(keys)
    # pylint: disable=protected-access
    result = ahmanalysis._group_observations(facade, keys.keys(), group_by="data_key")
    assert result == expected_result


@pytest.mark.parametrize(
    "obs_keys, expected_result",
    [
        pytest.param(
            ["obs_key"],
            {"obs_key": ["obs_key"]},
        ),
        pytest.param(
            ["obs_key_1", "obs_key_2"],
            {"obs_key_1": ["obs_key_1"], "obs_key_2": ["obs_key_2"]},
        ),
    ],
)
def test_group_obs_obs_key(obs_keys, expected_result):
    facade = MagicMock()
    # pylint: disable=protected-access
    result = ahmanalysis._group_observations(facade, obs_keys, group_by="obs_key")
    assert result == expected_result
    facade.assert_not_called()
