import numpy as np
import pandas as pd
import pytest

from semeio.workflows.correlated_observations_scaling.scaled_matrix import DataMatrix


def test_get_scaling_factor():
    np.random.seed(123)
    input_matrix = np.random.rand(10, 10)

    matrix = DataMatrix(pd.DataFrame(data=input_matrix))
    assert matrix.get_scaling_factor(10, 6) == np.sqrt(10 / 6.0)


@pytest.mark.parametrize(
    "threshold,expected_result", [(0.0, 1), (0.83, 4), (0.90, 5), (0.95, 6), (0.99, 7)]
)
def test_get_nr_primary_components(threshold, expected_result):
    np.random.seed(123)
    input_matrix = np.random.rand(10, 10)
    matrix = DataMatrix(pd.DataFrame(data=input_matrix))
    components, _ = matrix.get_nr_primary_components(threshold)
    assert components == expected_result


def test_std_normalization():
    input_matrix = pd.DataFrame(np.ones((3, 3)))
    input_matrix.loc["OBS"] = np.ones(3)
    input_matrix.loc["STD"] = np.ones(3) * 0.1
    expected_matrix = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    matrix = DataMatrix(pd.concat({"A_KEY": input_matrix}, axis=1))
    result = matrix.get_normalized_by_std()
    assert (result.loc[[0, 1, 2]].values == expected_matrix).all()
