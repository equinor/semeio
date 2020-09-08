import pytest

from semeio.workflows.correlated_observations_scaling import validator


@pytest.mark.parametrize(
    "test_input,expected_result",
    [
        (["POLY_OBS"], 0),
        (["NOT_A_KEY"], 1),
        (["POLY_OBS", "NOT_A_KEY"], 1),
        (["POLY_NOT", "NOT_A_KEY"], 2),
    ],
)
def test_has_keys(test_input, expected_result):
    obs = ["POLY_OBS"]
    msg = "fail_message"
    assert len(validator.has_keys(obs, test_input, msg)) == expected_result


@pytest.mark.parametrize(
    "input_list,result_list", [(["a"], []), (["a", "c"], []), (["a", "b", "c"], [])]
)
def test_is_subset_valid(input_list, result_list):
    example_list = ["a", "b", "c"]

    assert validator.is_subset(example_list, input_list) == result_list


@pytest.mark.parametrize(
    "input_list,list_length", [(["d"], 1), (["d", "e"], 2), (["a", "b", "d"], 1)]
)
def test_is_subset_invalid(input_list, list_length):
    example_list = ["a", "b", "c"]

    assert len(validator.is_subset(example_list, input_list)) == list_length
