# SparseLM: Sparse Linear Models

The goal of this package is to provide R and python interfaces to optimized C
implementations of commonly-used linear models (OLS, LASSO, logit) for sparse
model matrices.

Currently, it includes an optimized C implementation of the shooting algorithm
for LASSO with penalty loadings, as described in:

> Alexandre Belloni, Victor Chernozhukov, Christian Hansen. 2014. Inference
> on Treatment Effects after Selection among High-Dimensional Controls. The
> Review of Economic Studies 81(2): 608–650. https://doi.org/10.1093/restud/rdt044

Sequential coordinate descent has been replaced with randomized coordinate
descent to improve convergence. The C implementation uses the GNU Scientific
Library for its vector and sparse matrix operations and for interfacing with
BLAS. A dense implementation is also provided for reference.

The roadmap for additional features is:
* Expose R and python interfaces to the C implementation of LASSO.
* Replace the dense LU solver in the inner loop of the LASSO implementation
with a sparse solver.
* Implement OLS in C using singular value decomposition and the pseudoinverse,
for numerical stability.
* Use the singular value decomposition to invert the covariance matrix and
implement the sandwich estimator for robust standard errors.
* Investigate sparse implementations of logit.

## License

Copyright (C) 2017-2022 Mark Howison, Falmouth, ME.

SparseLM is available under the GNU General Public License v3.0.
See [LICENSE](https://github.com/mhowison/SparseLM/blob/main/LICENSE)
for more details.

Includes the PCG Random Number Generator, which is licensed under the Apache
License, Version 2.0.
