# CHANGELOG

## [1.18.0]

- Moved `fmudesign` from `fmu-tools` to `semeio` and made major improvements to the sampling engine as documentet below.

### Major Improvements in fmudesign's Sampling Engine

This version of `fmudesign` introduces major improvements to its sampling engine:

#### Executive Summary

1. Uses Latin Hypercube sampling instead of standard Monte Carlo sampling to provide more accurate representation of probability distributions with fewer samples. This method also produces more consistent results across repeated sampling runs.
2. Uses the Iman-Conover method to induce correlations while preserving marginal distributions. This approach requires Spearman rank correlations rather than Pearson correlations as inputs, which may be less intuitive for users.
3. Uses an improved algorithm for finding the nearest correlation matrix. Unlike the previous method, which could produce invalid results due to incorrect diagonal values, this new approach guarantees a mathematically valid correlation matrix.

These changes align `fmudesign`'s sampling capabilities more closely with industry-standard tools like Palisade @Risk.

Some more details are now given.

#### Latin Hypercube Sampling

`fmudesign` was originally designed to perform Monte Carlo Sampling (standard random sampling).
It allows users to specify correlations between parameters and uses Cholesky decomposition to induce these correlations.
This is the default approach used by `numpy`'s `multivariate_normal`, and it works well in many applications.
However, Monte Carlo Sampling does not reproduce distributions effectively when few samples are drawn.
In FMU simulations, which are computationally expensive, we need to limit the number of realizations to a minimum.
A standard solution to this limitation is to use Latin Hypercube Sampling (LHS) instead.

The image below shows histograms of 100 samples drawn using MC (left) and LHS (right),
demonstrating that samples drawn using LHS more closely resemble a uniform distribution.

<img src="img/mc_vs_lhs_uniform.png" alt="alt text" width="1000"/>

An additional benefit of using LHS with small samples is that resampling produces smaller variations in the sample distribution compared to MC.

Furthermore, we observe more stable convergence of the mean and variance.
The figure below shows the convergence of the mean of a simple non-linear function.
Notice that the LHS version achieves more stable results after approximately 30 samples.

<img src="img/convergence_of_mean_estimates.png" alt="alt text" width="1000"/>

#### Iman-Conover method

While replacing Monte Carlo with LHS addresses one issue, we still face the problem that direct Cholesky decomposition preserves neither marginal distributions nor the properties of sampling methods. @Risk solves this problem by implementing the Iman-Conover method, as described in their technical documentation:
https://www.uio.no/studier/emner/matnat/math/STK4400/v05/undervisningsmateriale/A%20distribution-free%20approach%20to%20rank%20correlation.pdf

We have followed the same approach and implemented our own version of the Iman-Conover method.

The figures below illustrate how the Iman-Conover method preserves marginal distributions while direct Cholesky does not.

<img src="img/mc_corr.png" alt="alt text" width="800"/>
<img src="img/lhs_corr.png" alt="alt text" width="800"/>

#### Nearest correlation matrix

Finally, we need a method to find the nearest correlation matrix to the one specified by users.
This is necessary because manually specifying a correlation matrix that satisfies all required mathematical properties can be challenging.
A valid correlation matrix must be positive semidefinite and have ones on the diagonal.

The current method implemented in fmudesign finds the nearest positive semidefinite matrix but does not constrain the diagonal to be all ones.
We replace this with a method described in the following paper (also implemented in Matlab's nearcorr):
https://ieeexplore.ieee.org/document/8160870

Previous version:

```
Input correlation matrix:
[[1.  0.9 0.  0. ]
 [0.9 1.  0.9 0. ]
 [0.  0.9 1.  0. ]
 [0.  0.  0.  1. ]]
Used closest positive semi-definite correlation matrix:
[[1.068 0.804 0.068 0.   ]
 [0.804 1.136 0.804 0.   ]
 [0.068 0.804 1.068 0.   ]
 [0.    0.    0.    1.   ]]
```

New version:

```
Warning: Correlation matrix is not consistent
Requirements:
  - Ones on the diagonal
  - Positive semi-definite matrix

Input correlation matrix:
[[1.00 0.90 0.00 0.00]
 [0.90 1.00 0.90 0.00]
 [0.00 0.90 1.00 0.00]
 [0.00 0.00 0.00 1.00]]

Adjusted to nearest consistent correlation matrix:
[[1.00 0.74 0.11 0.00]
 [0.74 1.00 0.74 0.00]
 [0.11 0.74 1.00 0.00]
 [0.00 0.00 0.00 1.00]]
```

#### Other changes

- Support for correlating discrete variables has been added
- Excel files produced by fmudesign now include a new sheet that displays the version of semeio used and the creation timestamp
