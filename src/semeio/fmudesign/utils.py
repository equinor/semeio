"""
Module for utility functions that do not belong elsewhere.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def is_consistent_correlation_matrix(matrix: npt.NDArray[Any]) -> bool:
    """
    Check if a matrix is a consistent correlation matrix.

    A correlation matrix is consistent if it has:
    1. All diagonal elements equal to 1
    2. Is positive semidefinite (all eigenvalues ≥ 0)

    Args:
        matrix: numpy array representing the correlation matrix

    Returns:
        bool: True if matrix is a consistent correlation matrix, False otherwise

    Examples
    --------
    >>> matrix = np.diag([1, 1, 1e-8])
    >>> is_consistent_correlation_matrix(matrix)
    False

    """
    # Check if diagonal elements are 1
    if not np.allclose(np.diagonal(matrix), 1):
        return False

    # Check positive semidefiniteness using eigenvalues
    try:
        eigenvals = np.linalg.eigvals(matrix)
        # Matrix is positive semidefinite if all eigenvalues are non-negative
        # Using small tolerance to account for numerical errors
        if not np.all(eigenvals > -1e-8):
            return False
    except np.linalg.LinAlgError:
        return False

    return True
