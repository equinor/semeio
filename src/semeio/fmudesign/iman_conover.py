"""
An implementation of the Iman-Conover transformation using hybrid Cholesky/SVD.

Using Iman-Conover with Latin Hybercube sampling
------------------------------------------------

Sample on the unit hypercube using LatinHypercube

>>> import scipy as sp
>>> sampler = sp.stats.qmc.LatinHypercube(d=2, seed=42, scramble=True)
>>> samples = sampler.random(n=100)

Map to distributions

>>> X = np.vstack((sp.stats.triang(0.5).ppf(samples[:, 0]),
...                sp.stats.gamma.ppf(samples[:, 1], a=1))).T

Induce correlations

>>> float(sp.stats.pearsonr(*X.T).statistic)
0.065898...
>>> correlation_matrix = np.array([[1, 0.3], [0.3, 1]])
>>> transform = ImanConover(correlation_matrix)
>>> X_transformed = transform(X)
>>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
0.279652...
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy as sp


def _is_positive_definite(X: npt.NDArray[Any]) -> bool:
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False


def _is_positive_semidefinite(X: npt.NDArray[np.float64]) -> bool:
    """Check if matrix is positive semidefinite using eigenvalue decomposition."""
    try:
        # PSD matrices must be square and symmetric
        if X.shape[0] != X.shape[1]:
            return False
        if not np.allclose(X, X.T, rtol=1e-12, atol=1e-12):
            return False

        eigenvals = np.linalg.eigvals(X)
        tol = 1e-12
        # Even though the eigenvalues of symmetric real matrices are guaranteed
        # to be real, they might turn out complex due to numerics.
        # Hence, the usage of np.real.
        return bool(np.all(np.real(eigenvals) >= -tol))
    except np.linalg.LinAlgError:
        return False


def _matrix_sqrt_with_pinv(
    X: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Compute matrix square root and its pseudoinverse using Cholesky if possible, otherwise SVD."""
    if _is_positive_definite(X):
        # Use Cholesky for positive definite matrices (faster and more stable)
        L = np.linalg.cholesky(X)
        # For Cholesky decomposition, inv(L) can be computed efficiently
        L_inv = sp.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        return L, L_inv.T
    else:
        # Fall back to SVD for positive semidefinite matrices
        U, s, _ = np.linalg.svd(X, hermitian=True)

        # Threshold small eigenvalues for numerical stability
        s_clipped = np.maximum(s, 0)
        s_sqrt = np.sqrt(s_clipped)

        # Keep all dimensions but zero out small eigenvalues
        tol = 1e-12
        mask = s_clipped >= tol
        s_sqrt[~mask] = 0

        # Square root matrix
        L = U * s_sqrt

        # Pseudoinverse of square root
        s_inv = np.zeros_like(s_sqrt)
        s_inv[mask] = 1.0 / s_sqrt[mask]
        L_pinv = (U * s_inv).T

        return L, L_pinv


class ImanConover:
    def __init__(self, correlation_matrix: npt.NDArray[Any]) -> None:
        """Create an Iman-Conover transform.

        Parameters
        ----------
        correlation_matrix : ndarray
            Target correlation matrix of shape (K, K).
            The Iman-Conover will try to induce a correlation on
            the data set X so that corr(X) is as close
            to `correlation_matrix` as possible.
            Can be positive definite or positive semidefinite.

        Notes
        -----
        The implementation follows the original paper:
        Iman, R. L., & Conover, W. J. (1982). A distribution-free approach to
        inducing rank correlation among input variables. Communications in
        Statistics - Simulation and Computation, 11(3), 311-334.
        https://www.tandfonline.com/doi/epdf/10.1080/03610918208812265?needAccess=true
        https://www.uio.no/studier/emner/matnat/math/STK4400/v05/undervisningsmateriale/A%20distribution-free%20approach%20to%20rank%20correlation.pdf

        Other useful sources:
        - https://blogs.sas.com/content/iml/2021/06/16/geometry-iman-conover-transformation.html
        - https://blogs.sas.com/content/iml/2021/06/14/simulate-iman-conover-transformation.html
        - https://aggregate.readthedocs.io/en/stable/5_technical_guides/5_x_iman_conover.html

        Examples
        --------
        Create a desired correction of 0.7 and a data set X with no correlation.
        >>> correlation_matrix = np.array([[1, 0.7], [0.7, 1]])
        >>> transform = ImanConover(correlation_matrix)
        >>> X = np.array([[0, 0  ],
        ...               [0, 0.5],
        ...               [0,  1 ],
        ...               [1, 0  ],
        ...               [1, 0.5],
        ...               [1, 1  ]])
        >>> X_transformed = transform(X)
        >>> X_transformed
        array([[0. , 0. ],
               [0. , 0. ],
               [0. , 0.5],
               [1. , 0.5],
               [1. , 1. ],
               [1. , 1. ]])

        The original data X has no correlation at all, while the transformed
        data has correlation that is closer to the desired correlation structure:

        >>> float(sp.stats.pearsonr(*X.T).statistic)
        0.0
        >>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
        0.816496...

        Achieving the exact correlation structure might be impossible. For the
        input matrix above, there is no permutation of the columns that yields
        the exact desired correlation of 0.7. Iman-Conover is a heuristic that
        tries to get as close as possible.

        With many samples, we get good results if the data are normal:

        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(size=(1000, 2))
        >>> X_transformed = transform(X)
        >>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
        0.697701...

        But if the data are far from normal (here:lognormal), the results are
        not as good. This is because correlation is induced in a normal space
        before the result is mapped back to the original marginal distributions.

        >>> rng = np.random.default_rng(42)
        >>> X = rng.lognormal(size=(1000, 2))
        >>> X_transformed = transform(X)
        >>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
        0.592541...
        """
        if not isinstance(correlation_matrix, np.ndarray):
            raise TypeError("Input argument `correlation_matrix` must be NumPy array.")
        if not correlation_matrix.ndim == 2:
            raise ValueError("Correlation matrix must be square.")
        if not correlation_matrix.shape[0] == correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if not np.allclose(np.diag(correlation_matrix), 1.0):
            raise ValueError("Correlation matrix must have 1.0 on diagonal.")
        if not np.allclose(correlation_matrix.T, correlation_matrix):
            raise ValueError("Correlation matrix must be symmetric.")
        if not _is_positive_semidefinite(correlation_matrix):
            raise ValueError("Correlation matrix must be positive semidefinite.")

        self.C = correlation_matrix.copy()

        # Use hybrid approach: Cholesky if possible, SVD otherwise
        self.P, _ = _matrix_sqrt_with_pinv(self.C)

    def __call__(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Transform an input matrix X.

        The output will have the same marginal distributions, but with
        induced correlation.

        Parameters
        ----------
        X : ndarray
            Input matrix of shape (N, K). This is the data set that we want to
            induce correlation structure on. X must have at least K + 1
            independent rows, because corr(X) cannot be singular.

        Returns
        -------
        ndarray
            Output matrix of shape (N, K). This data set will have a
            correlation structure that is more similar to `correlation_matrix`.

        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input argument `X` must be NumPy array.")
        if not X.ndim == 2:
            raise ValueError("Input matrix must be 2D.")

        N, K = X.shape

        if self.P.shape[0] != K:
            msg = f"Shape of `X` ({X.shape}) does not match shape of "
            msg += f"correlation matrix ({self.C.shape})"
            raise ValueError(msg)

        if N <= K:
            msg = f"The matrix X must have rows > columns. Got shape: {X.shape}"
            raise ValueError(msg)

        # STEP ONE - Use van der Waerden scores to transform data to
        # approximately multivariate normal (but with correlations).
        # The new data has the same rank correlation as the original data.
        ranks = sp.stats.rankdata(X, axis=0) / (N + 1)
        normal_scores = sp.stats.norm.ppf(ranks)

        # STEP TWO - Remove correlations from the transformed data
        empirical_correlation = np.corrcoef(normal_scores, rowvar=False)
        if not _is_positive_semidefinite(empirical_correlation):
            msg = "Rank data correlation not positive semidefinite."
            raise ValueError(msg)

        # Fail if input has perfect rank correlations that differ from target
        # (impossible to change perfect correlations via permutation)
        off_diagonal = empirical_correlation[~np.eye(K, dtype=bool)]
        if np.any(
            np.isclose(np.abs(off_diagonal), 1.0, rtol=1e-10)
        ) and not np.allclose(empirical_correlation, self.C, rtol=1e-6, atol=1e-6):
            msg = "Input data has perfect rank correlations that conflict with target correlation structure."
            raise ValueError(msg)

        # Use same hybrid approach for decorrelation
        decorrelation_matrix, decorrelation_pinv = _matrix_sqrt_with_pinv(
            empirical_correlation
        )

        if _is_positive_definite(empirical_correlation):
            # Use efficient triangular solve for Cholesky case
            # We exploit the fact that Q is lower-triangular and avoid the inverse.
            # X = N @ inv(Q)^T  =>  X @ Q^T = N  =>  (Q @ X^T)^T = N
            decorrelated_scores = sp.linalg.solve_triangular(
                decorrelation_matrix, normal_scores.T, lower=True
            ).T
        else:
            # Use pseudoinverse for SVD case
            decorrelated_scores = normal_scores @ decorrelation_pinv

        # STEP THREE - Induce correlations in transformed space
        correlated_scores = decorrelated_scores @ self.P.T

        # STEP FOUR - Map back to original space using ranks, ensuring
        # that marginal distributions are preserved
        result = np.empty_like(X)
        for k in range(K):
            # If row j is the k'th largest in `correlated_scores`, then
            # we map the k'th largest entry in X to row j.
            ranks = sp.stats.rankdata(correlated_scores[:, k]).astype(int) - 1
            result[:, k] = np.sort(X[:, k])[ranks]

        return result


def decorrelate(X: npt.NDArray[Any], remove_variance: bool = True) -> npt.NDArray[Any]:
    """Removes correlations or covariance from data X.

    Examples
    --------
    >>> X = np.array([[1. , 1. ],
    ...               [2. , 1.1],
    ...               [2.1, 3. ]])
    >>> X_decorr = decorrelate(X)
    >>> np.cov(X_decorr, rowvar=False).round(6)
    array([[1., 0.],
           [0., 1.]])
    >>> np.allclose(np.mean(X, axis=0), np.mean(X_decorr, axis=0))
    True

    >>> X_decorr = decorrelate(X, remove_variance=False)
    >>> np.cov(X_decorr, rowvar=False).round(6)
    array([[0.246667, 0.      ],
           [0.      , 0.846667]])
    >>> np.allclose(np.mean(X, axis=0), np.mean(X_decorr, axis=0))
    True
    """
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    cov = np.cov(X, rowvar=False)

    L = np.linalg.cholesky(cov)  # L @ L.T = cov
    if not remove_variance:
        L = L / np.sqrt(var)

    # Computes X = (X - mean) @ inv(L).T
    X = sp.linalg.solve_triangular(L, (X - mean).T, lower=True).T

    return mean + X


if __name__ == "__main__":
    import doctest

    doctest.testmod()
