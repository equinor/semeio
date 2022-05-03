import pytest

from semeio.workflows.spearman_correlation_job import job as spearman


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        ([(1, "KEY_1", 1)], {1: {"KEY_1": [1]}}),
        ([(1, "KEY_1", 1), (1, "KEY_1", 2)], {1: {"KEY_1": [1, 2]}}),
        ([(1, "KEY_1", 1), (2, "KEY_2", 2)], {1: {"KEY_1": [1]}, 2: {"KEY_2": [2]}}),
    ],
)
def test_make_clusters(test_input, expected_result):
    result = spearman._cluster_data(test_input)
    assert result == expected_result


@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        ({}, {}),
        ({1: {"KEY_1": [1]}}, {}),
        ({1: {"KEY_1": [1, 2]}}, {0: {"KEY_1": [1, 2]}}),
        (
            {1: {"KEY_1": [1], "KEY_2": [2]}, 2: {"KEY_2": [2]}},
            {0: {"KEY_1": [1], "KEY_2": [2]}},
        ),
    ],
)
def test_single_obs_clusters(test_input, expected_result):
    result = spearman._remove_singular_obs(test_input)
    assert result == expected_result, "bad removal of singular obs clusters"


@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        (
            {1: {"KEY_1": [1]}},
            [{"CALCULATE_KEYS": {"keys": [{"key": "KEY_1", "index": [1]}]}}],
        )
    ],
)
def test_config_creation(test_input, expected_result):
    result = spearman._config_creation(test_input)
    assert result == expected_result
