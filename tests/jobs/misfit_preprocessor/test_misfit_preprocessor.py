import numpy as np
import pandas as pd
import pytest
import random
from semeio.workflows import misfit_preprocessor

from unittest.mock import Mock


class MockedMeasuredData(object):
    def __init__(self, observations, responses):
        self._data = self._build_data(observations, responses)

    def _build_data(self, observations, responses):
        def assert_equal_keys(a, b):
            assert sorted(a.keys()) == sorted(b.keys())

        columns = []
        ens_size = None
        assert_equal_keys(observations, responses)
        for key in observations.keys():
            assert_equal_keys(observations[key], responses[key])
            for sub_key, sim in responses[key].items():
                if ens_size is None:
                    ens_size = len(sim)
                assert ens_size == len(sim)
                columns.append((key,) + tuple(sub_key))

        index = ("OBS", "STD") + tuple(range(ens_size))

        raw_data = np.array(
            [
                np.array([observations[col[0]][col[1:]][0] for col in columns]),
                np.array([observations[col[0]][col[1:]][1] for col in columns]),
            ]
        )
        simulated_raw_data = np.array([responses[col[0]][col[1:]] for col in columns]).T
        raw_data = np.concatenate((raw_data, simulated_raw_data))

        data = pd.DataFrame(
            raw_data,
            index=index,
            columns=pd.MultiIndex.from_tuples(
                columns, names=[None, "key_index", "data_index"]
            ),
        )

        return data

    @property
    def data(self):
        return self._data

    def get_simulated_data(self):
        return self._data.drop(index=["OBS", "STD"])

    def remove_failed_realizations(self):
        pass

    def remove_inactive_observations(self):
        pass

    def filter_ensemble_std(self, _):
        pass


def assert_homogen_clusters(config):
    for sub_config in config:
        assert 1 == len(sub_config["CALCULATE_KEYS"]["keys"])


def generate_simulated_responses(
    forward_polynomials,
    parameter_distribution,
    poly_states,
    ensemble_size,
):
    simulated = {}
    for poly_idx, (poly_fm, states) in enumerate(zip(forward_polynomials, poly_states)):
        new_parameters = np.random.uniform(0, 10, 3 * ensemble_size)
        new_parameters.resize(ensemble_size, 3)
        simulated["poly_{}".format(poly_idx)] = {
            (state, state): sum((new_parameters * np.array((state ** 2, state, 1))).T)
            for state in states
        }

    return simulated


def generate_observations(
    forward_polynomials,
    parameter_distribution,
    poly_states,
):
    true_parameters = parameter_distribution()

    return {
        "poly_{}".format(poly_idx): {
            (state, state): (
                poly_fm(
                    a=true_parameters[poly_idx]["a"],
                    b=true_parameters[poly_idx]["b"],
                    c=true_parameters[poly_idx]["c"],
                    x=state,
                ),
                1,
            )
            for state in states
        }
        for poly_idx, (poly_fm, states) in enumerate(
            zip(forward_polynomials, poly_states)
        )
    }


def generate_measurements(num_polynomials, poly_states=None, ensemble_size=10000):
    if poly_states is None:
        poly_states = [range(3) for _ in range(num_polynomials)]

    forward_polynomials = [
        lambda a, b, c, x: a * x ** 2 + b * x + c for _ in range(num_polynomials)
    ]

    def parameter_distribution():
        return [
            {
                "a": random.uniform(0, 10),
                "b": random.uniform(0, 10),
                "c": random.uniform(0, 10),
            }
            for _ in enumerate(forward_polynomials)
        ]

    observations = generate_observations(
        forward_polynomials,
        parameter_distribution,
        poly_states,
    )
    simulated = generate_simulated_responses(
        forward_polynomials,
        parameter_distribution,
        poly_states,
        ensemble_size,
    )
    return observations, simulated


@pytest.mark.parametrize("method", ["spearman_correlation", "auto_scale"])
@pytest.mark.parametrize(
    "num_polynomials",
    tuple(range(1, 5)) + (20, 100),
)
def test_misfit_preprocessor_n_polynomials(num_polynomials, method):
    """
    The goal of this test is to create a data set of uncorrelated polynomials,
    meaning that there should be as many clusters as there are input polynomials.
    """
    state_size = 3
    poly_states = [range(1, state_size + 1) for _ in range(num_polynomials)]

    observations, simulated = generate_measurements(
        num_polynomials,
        poly_states=poly_states,
        ensemble_size=10000,
    )
    measured_data = MockedMeasuredData(observations, simulated)
    # We set the PCA threshold to 0.99 so a high degree of correlation is required
    # to have an impact. Setting it this way only has an impact for "auto_scale"
    config = {"clustering": {"method": method}, "scaling": {"threshold": 0.99}}
    reporter_mock = Mock()
    configs = misfit_preprocessor.run(config, measured_data, reporter_mock)
    assert_homogen_clusters(configs)
    assert num_polynomials == len(configs), configs


@pytest.mark.parametrize("linkage", ["average", "single"])
@pytest.mark.parametrize("method", ["spearman_correlation", "auto_scale"])
@pytest.mark.parametrize(
    "state_size",
    [5 * [30], [5, 5, 5, 5, 100]],
)
def test_misfit_preprocessor_state_size(state_size, method, linkage):
    if state_size == [5, 5, 5, 5, 100]:
        if linkage == "average":
            pytest.skip("Produces wrong number of clusters")
        elif method == "auto_scale":
            pytest.skip("Produces not homogeneous clusters due to PCA analysis")

    num_polynomials = 5
    poly_states = [range(1, size + 1) for size in state_size]

    observations, simulated = generate_measurements(
        num_polynomials,
        poly_states=poly_states,
        ensemble_size=30000,
    )
    measured_data = MockedMeasuredData(observations, simulated)

    config = {
        "clustering": {"method": method, method: {"linkage": {"method": linkage}}},
        "scaling": {"threshold": 0.99},
    }
    reporter_mock = Mock()
    configs = misfit_preprocessor.run(config, measured_data, reporter_mock)
    assert_homogen_clusters(configs)
    assert num_polynomials == len(configs), configs


@pytest.mark.parametrize(
    "state_size",
    [5 * [30], [5, 5, 5, 5, 100], [5, 5, 5, 5, 100], [2, 1000]],
)
def test_misfit_preprocessor_state_uneven_size(state_size):
    num_polynomials = len(state_size)
    poly_states = [range(1, size + 1) for size in state_size]

    observations, simulated = generate_measurements(
        num_polynomials,
        poly_states=poly_states,
        ensemble_size=30000,
    )
    measured_data = MockedMeasuredData(observations, simulated)

    config = {
        "clustering": {
            "method": "spearman_correlation",
            "spearman_correlation": {
                "fcluster": {"t": num_polynomials + 1, "criterion": "maxclust"}
            },
        }
    }
    reporter_mock = Mock()
    configs = misfit_preprocessor.run(config, measured_data, reporter_mock)
    assert num_polynomials == len(configs), configs
    assert_homogen_clusters(configs)


def test_misfit_preprocessor_configuration_errors():
    observations, simulated = generate_measurements(1)
    measured_data = MockedMeasuredData(observations, simulated)

    config = {
        "unknown_key": [],
        "clustering": {
            "method": "spearman_correlation",
            "spearman_correlation": {"fcluster": {"threshold": 1.0}},
        },
    }
    reporter_mock = Mock()
    with pytest.raises(misfit_preprocessor.ValidationError) as ve:
        misfit_preprocessor.run(config, measured_data, reporter_mock)

    expected_err_msg = (
        "Invalid configuration of misfit preprocessor\n"
        "  - Unknown key: unknown_key (root level)\n"
        "  - Unknown key: threshold (clustering.spearman_correlation.fcluster)\n"
    )
    assert expected_err_msg == str(ve.value)


@pytest.mark.parametrize(
    "num_polynomials",
    tuple(range(2, 5)) + (20, 100),
)
def test_misfit_preprocessor_n_polynomials_w_correlation(num_polynomials):
    state_size = 3
    poly_states = [range(1, state_size + 1) for _ in range(num_polynomials)]

    observations, simulated = generate_measurements(
        num_polynomials,
        poly_states=poly_states,
        ensemble_size=10000,
    )
    measured_data = MockedMeasuredData(observations, simulated)

    # We add a correlation:
    measured_data.data["poly_0"] = measured_data.data["poly_1"] * 2.0

    config = {
        "clustering": {"method": "spearman_correlation"},
        "scaling": {"threshold": 0.99},
    }
    reporter_mock = Mock()
    configs = misfit_preprocessor.run(config, measured_data, reporter_mock)
    assert num_polynomials == len(configs) - 1, configs
