"""
An implementation of the Iman-Conover transformation.

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

>>> sp.stats.pearsonr(*X.T).statistic
0.065898...
>>> correlation_matrix = np.array([[1, 0.3], [0.3, 1]])
>>> transform = ImanConover(correlation_matrix)
>>> X_transformed = transform(X)
>>> sp.stats.pearsonr(*X_transformed.T).statistic
0.279652...
"""

import numpy as np
import scipy as sp


def _is_positive_definite(X):
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False


class ImanConover:
    def __init__(self, correlation_matrix):
        """Create an Iman-Conover transform.

        Parameters
        ----------
        correlation_matrix : ndarray
            Target correlation matrix of shape (K, K). The Iman-Conover will
            try to induce a correlation on the data set X so that corr(X) is
            as close to `correlation_matrix` as possible.

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

        >>> sp.stats.pearsonr(*X.T).statistic
        0.0
        >>> sp.stats.pearsonr(*X_transformed.T).statistic
        0.816496...

        Achieving the exact correlation structure might be impossible. For the
        input matrix above, there is no permutation of the columns that yields
        the exact desired correlation of 0.7. Iman-Conover is a heuristic that
        tries to get as close as possible.

        With many samples, we get good results if the data are normal:

        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(size=(1000, 2))
        >>> X_transformed = transform(X)
        >>> sp.stats.pearsonr(*X_transformed.T).statistic
        0.697701...

        But if the data are far from normal (here:lognormal), the results are
        not as good. This is because correlation is induced in a normal space
        before the result is mapped back to the original marginal distributions.

        >>> rng = np.random.default_rng(42)
        >>> X = rng.lognormal(size=(1000, 2))
        >>> X_transformed = transform(X)
        >>> sp.stats.pearsonr(*X_transformed.T).statistic
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
        if not _is_positive_definite(correlation_matrix):
            raise ValueError("Correlation matrix must be positive definite.")

        self.C = correlation_matrix.copy()
        self.P = np.linalg.cholesky(self.C)

    def __call__(self, X):
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
            raise ValueError("Correlation matrix must be square.")

        N, K = X.shape

        if self.P.shape[0] != K:
            msg = f"Shape of `X` ({X.shape}) does not match shape of "
            msg += f"correlation matrix ({self.P.shape})"
            raise ValueError(msg)

        if N <= K:
            msg = f"The matrix X must have rows > columns. Got shape: {X.shape}"
            raise ValueError(msg)

        # STEP ONE - Use van der Waerden scores to transform data to
        # approximately multivariate normal (but with correlations).
        # The new data has the same rank correlation as the original data.
        ranks = sp.stats.rankdata(X, axis=0) / (N + 1)
        normal_scores = sp.stats.norm.ppf(ranks)  # + np.random.randn(N, K) * epsilon

        # STEP TWO - Remove correlations from the transformed data
        empirical_correlation = np.corrcoef(normal_scores, rowvar=False)
        if not _is_positive_definite(empirical_correlation):
            msg = "Rank data correlation not positive definite."
            msg += "There are perfect correlations in the ranked data."
            msg += "Supply more data (rows in X) or sample differently."
            raise ValueError(msg)

        decorrelation_matrix = np.linalg.cholesky(empirical_correlation)

        # We exploit the fact that Q is lower-triangular and avoid the inverse.
        # X = N @ inv(Q)^T  =>  X @ Q^T = N  =>  (Q @ X^T)^T = N
        decorrelated_scores = sp.linalg.solve_triangular(
            decorrelation_matrix, normal_scores.T, lower=True
        ).T

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


def decorrelate(X, remove_variance=True):
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


import itertools


class PermutationCorrelator:
    def __init__(
        self,
        correlation_matrix,
        *,
        weights=None,
        iterations=1000,
        max_iter_no_change=250,
        correlation_type="pearson",
        seed=None,
        verbose=False,
    ):
        """


        Parameters
        ----------
        correlation_matrix : 2d numpy array
            A target correlation matrix.
        weights : 2d numpy array or None, optional
            Elementwise weights for the target correlation matrix.
            The default is None, which corresponds to uniform weights.
        iterations : int, optional
            Maximal number of iterations to run. Each iterations consists of
            one loop over all variables. Choosing 0 means infinite iterations.
            The default is 1000.
        max_iter_no_change : int, optional
            Maximal number of iterations without any improvement in the
            objective before terminating. The default is 250.
        correlation_type : str, optional
            Either "pearson" or "spearman". The default is "pearson".
        seed : int or None, optional
            A seed for the random number generator. The default is None.
        verbose : bool, optional
            Whether or not to print information. The default is False.

        Examples
        --------
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(size=(100, 2))
        >>> sp.stats.pearsonr(*X.T).statistic
        0.1573...
        >>> correlation_matrix = np.array([[1, 0.7], [0.7, 1]])
        >>> perm_trans = PermutationCorrelator(correlation_matrix, seed=0)
        >>> X_transformed = perm_trans.hill_climb(X)
        >>> sp.stats.pearsonr(*X_transformed.T).statistic
        0.6999...

        For large matrices, it often makes sense to first use Iman-Conover
        to get a good initial solution, then give it to PermutationCorrelator.
        Start by creating a large correlation matrix:

        >>> variables = 25
        >>> correlation_matrix = np.ones((variables, variables)) * 0.7
        >>> np.fill_diagonal(correlation_matrix, 1.0)
        >>> perm_trans = PermutationCorrelator(correlation_matrix,
        ...                                    iterations=1000,
        ...                                    max_iter_no_change=250,
        ...                                    seed=0, verbose=True)

        Create data X, then transform using Iman-Conover:

        >>> X = rng.normal(size=(10 * variables, variables))
        >>> perm_trans._error(X) # Initial error
        0.4846...
        >>> ic_trans = ImanConover(correlation_matrix)
        >>> X_ic = ic_trans(X)
        >>> perm_trans._error(X_ic) # Error after Iman-Conover
        0.0071...
        >>> X_ic_pc = perm_trans.hill_climb(X_ic)
        Running permutation correlator for 1000 iterations.
         Iteration 100  Error: 0.0037
         Iteration 200  Error: 0.0024
         Iteration 300  Error: 0.0017
         Iteration 400  Error: 0.0014
         Iteration 500  Error: 0.0012
         Iteration 600  Error: 0.0010
         Iteration 700  Error: 0.0008
         Iteration 800  Error: 0.0007
         Iteration 900  Error: 0.0006
         Terminating at iteration 940.
         No improvement for 250 iterations. Error: 0.0006
        >>> perm_trans._error(X_ic_pc) # Error after Iman-Conover + permutation
        0.0006119...
        """
        corr_types = {"pearson": self._pearson, "spearman": self._spearman}

        # Validate all input arguments
        if not isinstance(correlation_matrix, np.ndarray):
            raise TypeError("Input argument `correlation_matrix` must be NumPy array.")
        if not correlation_matrix.ndim == 2:
            raise ValueError("Correlation matrix must be square.")
        if not correlation_matrix.shape[0] == correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if not np.allclose(correlation_matrix.T, correlation_matrix):
            raise ValueError("Correlation matrix must be symmetric.")
        if not (weights is None or (isinstance(weights, np.ndarray))):
            raise TypeError("Input argument `weights` must be None or NumPy array.")
        if not (weights is None or weights.shape == correlation_matrix.shape):
            raise ValueError("`weights` and `correlation_matrix` must have same shape.")
        if not (weights is None or np.all(weights > 0)):
            raise ValueError("`weights` must have positive entries.")
        if not isinstance(iterations, int) and iterations >= 0:
            raise ValueError("`iterations` must be non-negative integer.")
        if not isinstance(max_iter_no_change, int) and max_iter_no_change >= 0:
            raise ValueError("`max_iter_no_change` must be non-negative integer.")
        if iterations == 0 and max_iter_no_change == 0:
            raise ValueError("`iterations` or `max_iter_no_change` must be positive.")
        if not (isinstance(correlation_type, str) and correlation_type in corr_types):
            raise ValueError(
                f"`correlation_type` must be one of: {tuple(corr_types.keys())}"
            )
        if not (seed is None or isinstance(seed, int)):
            raise TypeError("`seed` must be None or an integer")
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be boolean")

        self.C = correlation_matrix.copy()
        weights = np.ones_like(self.C) if weights is None else weights
        self.weights = weights / np.sum(weights)
        self.iters = iterations
        self.max_iter_no_change = max_iter_no_change
        self.correlation_func = corr_types[correlation_type]
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.triu_indices = np.triu_indices(self.C.shape[0], k=1)

    def _pearson(self, X):
        """Given a matrix X of shape (m, n), return a matrix of shape (n, n)
        with Pearson correlation coefficients."""
        return np.corrcoef(X, rowvar=False)

    def _spearman(self, X):
        """Given a matrix X of shape (m, n), return a matrix of shape (n, n)
        with Spearman correlation coefficients."""
        if X.shape[1] == 2:
            spearman_corr = sp.stats.spearmanr(X).statistic
            return np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])
        else:
            return sp.stats.spearmanr(X).statistic

    @staticmethod
    def _swap(X, i, j, k):
        """Swap rows i and j in column k inplace."""
        X[i, k], X[j, k] = X[j, k], X[i, k]

    def _error(self, X):
        """Compute RMSE over upper triangular part of corr(X) - C."""
        corr = self.correlation_func(X)  # Correlation matrix
        idx = self.triu_indices  # Get upper triangular indices (ignore diag)
        weighted_residuals_sq = self.weights[idx] * (corr[idx] - self.C[idx]) ** 2.0
        return np.sqrt(np.sum(weighted_residuals_sq))

    def hill_climb(self, X):
        """Hill climbing swaps two random rows (observations). If the result
        leads to a smaller error, then it is kept. If not we try again."""

        num_obs, num_vars = X.shape
        if self.verbose:
            print(f"Running permutation correlator for {self.iters} iterations.")

        # Set up loop generator
        iters = range(1, self.iters + 1) if self.iters else itertools.count(1)
        loop_gen = itertools.product(iters, range(num_vars))  # (iteration, k)

        # Set up variables that are tracked in the main loop
        current_X = X.copy()
        current_error = self._error(current_X)
        iter_no_change = 0

        # Main loop. For each iteration, k cycles through all variables
        for iteration, k in loop_gen:
            print_iter = iteration % (self.iters // 10) if self.iters else 1000
            if self.verbose and print_iter == 0 and k == 0:
                print(f" Iteration {iteration}  Error: {current_error:.4f}")

            # Choose a random variable and two random observations
            # Using rng.integers() is faster than rng.choice(replace=False)
            i, j = self.rng.integers(0, high=num_obs, size=2)
            if i == j:
                continue

            # Turn current_X into a new proposed X by swapping two observations
            # i and j in column (variable) k.
            self._swap(current_X, i, j, k)
            proposed_error = self._error(current_X)

            # The proposed X was better
            if proposed_error < current_error:
                current_error = proposed_error
                iter_no_change = 0

            # The proposed X was worse
            else:
                self._swap(current_X, i, j, k)  # Swap indices back
                iter_no_change += 1

            # Termination condition triggered
            if self.max_iter_no_change and (iter_no_change >= self.max_iter_no_change):
                if self.verbose:
                    print(f""" Terminating at iteration {iteration}.
 No improvement for {iter_no_change} iterations. Error: {current_error:.4f}""")
                break

        return current_X


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "--doctest-modules"])

if __name__ == "__main__" and True:
    import matplotlib.pyplot as plt

    # Create data
    sampler = sp.stats.qmc.LatinHypercube(d=2, seed=42, scramble=True)
    X = sampler.random(n=1000)
    correlation_matrix = np.array([[1, 0.7], [0.7, 1]])

    # IC only
    transform = ImanConover(correlation_matrix)
    X_ic = transform(X)
    IC_corr = sp.stats.pearsonr(*X_ic.T).statistic
    print("IC", IC_corr)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 3))
    ax1.set_title(f"IC with correlation: {IC_corr:.3f}")
    ax1.scatter(*X_ic.T, s=1)

    # PC only
    iterations = 500  # int(X.shape[0]**1.5)
    pc_transform = PermutationCorrelator(
        correlation_matrix,
        seed=0,
        iterations=iterations,
        max_iter_no_change=100,
        verbose=True,
    )
    X_pc = pc_transform.hill_climb(X)
    PC_corr = sp.stats.pearsonr(*X_pc.T).statistic
    print("PC", PC_corr)

    ax2.set_title(f"HC: {PC_corr:.3f}")
    ax2.scatter(*X_pc.T, s=1)

    # IC first, then PC on the results
    X_ic_pc = pc_transform.hill_climb(X_ic)
    IC_PC_corr = sp.stats.pearsonr(*X_ic_pc.T).statistic
    print("IC + PC", IC_PC_corr)

    ax3.set_title(f"IC + HC: {IC_PC_corr:.3f}")
    ax3.scatter(*X_ic_pc.T, s=1)

    fig.tight_layout()
    plt.show()
