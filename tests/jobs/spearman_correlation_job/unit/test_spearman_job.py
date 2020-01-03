# -*- coding: utf-8 -*-
import sys

import pytest
from semeio.jobs.spearman_correlation_job import job as spearman

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


@pytest.mark.parametrize(
    "test_input",
    [
        ([{"CALCULATE_KEYS": {"keys": [{"key": "KEY_1", "index": [1]}]}}]),
        ([{"CALCULATE_KEYS": {"keys": [{"key": "KEY_1", "index": [1, 2]}]}}]),
        (
            [
                {
                    "CALCULATE_KEYS": {
                        "keys": [
                            {"key": "KEY_1", "index": [1, 2]},
                            {"key": "KEY_2", "index": [1, 2]},
                        ]
                    }
                }
            ]
        ),
        (
            [
                {"CALCULATE_KEYS": {"keys": [{"key": "KEY_1", "index": [1, 2]}]}},
                {"CALCULATE_KEYS": {"keys": [{"key": "KEY_2", "index": [5, 6]}]}},
            ]
        ),
    ],
)
def test_run_scaling(test_input, monkeypatch):
    facade = Mock()
    scal_job = Mock()
    monkeypatch.setattr(spearman, "scaling_job", scal_job)
    spearman._run_scaling(facade, test_input)
    for input_config in test_input:
        assert scal_job.any_call(input_config)
    assert scal_job.call_count == len(test_input)


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
        (
            {1: {"KEY_1": [1]}},
            [{"CALCULATE_KEYS": {"keys": [{"key": "KEY_1", "index": [1]}]}}],
        )
    ],
)
def test_config_creation(test_input, expected_result):
    result = spearman._config_creation(test_input)
    assert result == expected_result
