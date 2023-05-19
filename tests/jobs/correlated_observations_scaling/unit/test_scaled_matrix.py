import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from semeio.workflows.correlated_observations_scaling.scaled_matrix import DataMatrix

rng = np.random.default_rng()


def test_get_scaling_factor():
    np.random.seed(123)
    input_matrix = np.random.rand(10, 10)

    matrix = DataMatrix(pd.DataFrame(data=input_matrix))
    assert matrix.get_scaling_factor(10, 6) == np.sqrt(10 / 6.0)


@pytest.mark.parametrize(
    "threshold,expected_result", [(0.0, 0), (0.83, 3), (0.90, 4), (0.95, 5), (0.99, 6)]
)
def test_get_nr_primary_components(threshold, expected_result):
    np.random.seed(123)
    input_matrix = np.random.rand(10, 10)
    matrix = DataMatrix(pd.DataFrame(data=input_matrix))
    components, _ = matrix.get_nr_primary_components(threshold)
    assert components == expected_result


def test_that_get_nr_primary_components_is_according_to_theory():
    # pylint: disable=too-many-locals,invalid-name
    """Based on theory in Multivariate Statistical Methods 4th Edition
    by Donald F. Morrison.
    See section 6.5 - Some Patterned Matrices and Their Principal Components.
    """
    rho = 0.3
    p = 4
    sigma = 1
    N = 100000

    R = np.ones(shape=(p, p)) * rho
    np.fill_diagonal(R, sigma**2)

    # Fast sampling of correlated multivariate observations
    X = rng.standard_normal(size=(p, N))
    Y = (np.linalg.cholesky(R) @ X).T

    Y = StandardScaler().fit_transform(Y)

    lambda_1 = sigma**2 * (1 + (p - 1) * rho)
    lambda_remaining = sigma**2 * (1 - rho)
    s1 = np.sqrt(lambda_1 * (N - 1))
    s_remaining = np.sqrt(lambda_remaining * (N - 1))

    total = s1**2 + (p - 1) * s_remaining**2
    threshold_1 = s1**2 / total
    threshold_2 = (s1**2 + s_remaining**2) / total
    threshold_3 = (s1**2 + 2 * s_remaining**2) / total

    matrix = DataMatrix(pd.DataFrame(data=Y))

    # Adding a bit to the thresholds because of numerical accuracy.
    components, _ = matrix.get_nr_primary_components(threshold_1 + 0.01)
    assert components == 1
    components, _ = matrix.get_nr_primary_components(threshold_2 + 0.01)
    assert components == 2
    components, _ = matrix.get_nr_primary_components(threshold_3 + 0.01)
    assert components == 3


def test_std_normalization():
    input_matrix = pd.DataFrame(np.ones((3, 3)))
    input_matrix.loc["OBS"] = np.ones(3)
    input_matrix.loc["STD"] = np.ones(3) * 0.1
    expected_matrix = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    matrix = DataMatrix(pd.concat({"A_KEY": input_matrix}, axis=1))
    result = matrix.get_normalized_by_std()
    assert (result.loc[[0, 1, 2]].values == expected_matrix).all()
