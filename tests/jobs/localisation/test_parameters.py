import pytest

from semeio.workflows.localisation.local_script_lib import Parameters


@pytest.mark.parametrize(
    "input_list", (["A", "B:C"], ["A", "C"], ["A:1", "A:2", "A:3", "C"])
)
def test_from_list(input_list):
    parameters = Parameters.from_list(input_list)
    assert parameters.to_list() == input_list


@pytest.mark.parametrize(
    "input_list, expected_error",
    (
        [["A:B:C"], "Too many : in A:B:C"],
        (["A", "A:1"], "did not expect parameters, found 1"),
        (["A:1", "A"], r"found A in {'A': \['1'\]}, but did not find parameters"),
    ),
)
def test_from_list_error(input_list, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        Parameters.from_list(input_list)
