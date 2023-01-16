"""
SparseLM: Sparse Linear Models
"""

import numpy as np
from scipy import sparse, stats

def _se_classical(e, n, m, pinv):
    """
    Compute classical standard errors from the sum of squared errors and
    the diagnoal of the variance-covariance matrix.

    For efficiency, the diagonal is directly calculated from the columns
    of the pseudoinverse `pinv`, instead of the entire matrix product
    pinv pinv^T
    """
    sse = np.multiply(e, e).sum() / (n - m)
    # For efficiency, directly calculate the diagonal instead of the entire
    # matrix product pinv * pinv.T
    vcov = [
        np.dot(pinv[:,i], pinv[:,i])
        for i in range(m)
    ]
    return np.sqrt(sse * np.array(vcov))

def _se_robust(e, n, m, pinv, robust="HC1"):
    """
    Estimate heteroskedastic-robust clustered standard errors using either the
    HC0 or HC1 method with the sandwhich estiamtor:

    vcov = (X^T X)^{-1} X^T e e^T X (X^T X)^{-1}
    se = sqrt(diag(vcov))

    For efficiency, the sandwich estimator is split into two halves and
    the diagonal calculated directly, without the entire matrix product,
    using the pseudoinverse:

    vcov1 = pinv e
    vcov2 = e^T pinv^T = vcov1^T
    se = sqrt(vcov1 vcov2)
    """
    vcov1 = pinv * e
    vcov2 = vcov1.T
    if robust == "HC0":
        c = 1 # HC0 is the sandwich estimator without any further corrections
    if robust == "HC1":
        c = n / (n - m)
    else:
        raise ValueError(f"unknown robust type '{robust}'")
    return np.sqrt(c * np.multiply(vcov1, vcov2))

def _se_clustered_robust(e, n, m, pinv, clustered, robust="HC1"):
    """
    Estimate heteroskedastic-robust clustered standard errors using
    a block diagonal plug-in matrix B to the sandwich estimator, where
    the blocks are defined by the cluster membership of the individual
    observations:

    B = e e^T
    vcov = (X^T X)^{-1} X^T B X (X^T X)^{-1}
    se = sqrt(diag(vcov))

    The estimation strategy is to precompute the LHS and RHS of the sandwich
    estimator:

    LHS = (X^T X)^{-1} X^T
    RHS = X (X^T X)^{-1}

    In the context of an SVD solution to the least-squares problem, these are
    equal to the pseudoinverse and the transpose of the pseudoinverse.

    Because only the the diagonal of the variance-covariance matrix is used to
    calculate the standard errors, we can estimate the separate halves of the
    sandwich estimator and multiply the resulting column vector and row vector
    to directly calculate the diagonal vector:

    vcov1 = pinv e
    vcov2 = e^T pinv^T = vcov1^T
    se = sqrt(vcov1 vcov2)

    The vcov1/vcov2 vectors are aggregated iteratively over the clusters.
    """
    vcov1 = np.zeros(m)
    for idx in clustered:
        for i in idx: # cluster indices are column indices in pinv (csc)
            ei = e[i]
            i0 = pinv.indptr[i]
            i1 = pinv.indptr[i+1]
            for j, x in zip(pinv.indices[i0:i1], pinv.data[i0:i1]):
                vcov1[j] += (x * ei)
    vcov2 = vcov1.T
    # Correction for finite number of clusters
    c = len(clustered)
    c = (c / (c - 1)) * ((n - 1) / (m - 1))
    if robust == "HC0":
        pass # HC0 is the sandwich estimator without any further corrections
    if robust == "HC1":
        c *= n / (n - m)
    else:
        raise ValueError(f"unknown robust type '{robust}'")
    return np.sqrt(c * np.multiply(vcov1, vcov2))


class OLS():
    """
    Ordinary Least Squares regression.
    
    `X` is a sparse design matrix in compressed sparse column (CSC) format.
    `y` is a sparse or dense array.

    No intercept will be added by default. Specify `intercept=True` to add an intercept.
    (Note that X will be temporarily modified to append a constant column, which will be
    removed before the constructor returns.)

    Robust (heteroskedastic-consistent) standard errors can be calculated by specifying an
    HC type for `robust`:

    "HC0" is the original White's (1980) sandwich estimator
    "HC1" is MacKinnon and White's (1985) adjustment to correct for degrees of freedom
    (HC2-HC5 are not yet implemented)

    Standard errors can be clustered by an array of cluster indices in `clustered`.
    """

    def __init__(self, X, y, intercept=False, varnames=None, robust=None, clustered=None):

        # Argument checks
        if not sparse.isspmatrix_csc(X):
            raise ValueError("X must be a sparse design matrix in compressed sparse column (CSC) format")
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same length")
        if varnames is not None and len(varnames) != X.shape[1]:
            raise ValueError("X and varnames must have same width")

        n = X.shape[0]
        m = X.shape[1]

        # Add intercept
        if intercept:
            X = sparse.hstack([X, np.ones(n)])

        # Singular value decomposition to calculate the condition number and pseudoinverse
        U, sv, V_T = svd(X) # TODO: identify a performant sparse SVD library
        self.cond = sv.max() / sv.min() # condition number of the design matrix X is the ratio of the max/min singular values
        psv = np.where(sv == 0, 0, 1.0 / sv) # missing singular values are replaced with 0 instead of their (undefined) inverse 
        pinv = V_T.T * sparse.diags(psv) * U.T
        assert sparse.isspmatrix_csc(pinv)

        # Calculate beta
        beta = pinv * y
        if intercept:
            self.intercept = beta[-1]
            self.beta = beta[:-1]
        else:
            self.beta = beta

        # Predict y and calculate residuals
        self.yhat = self.predict(X)
        self.resids = y - self.yhat

        # Remove intercept
        if intercept:
            X = X[:,:m]
            pinv = pinv[:m,:m]

        # Estimate standard errors
        if clustered is not None:
            if robust is not None:
                self.se = _se_clustered_robust(self.resids, n, m, pinv, clustered, robust)
            else:
                self.se = _se_clustered_robust(self.resids, n, m, pinv, clustered)
        elif robust is not None:
            self.se = _se_robust(self.resids, n, m, pinv, robust)
        else:
            self.se = _se_classical(self.resids, n, m, pinv)

        # Calculate t-statistics and p-values
        self.t = self.beta / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), n - m))


    def predict(self, X):
        """
        Predict `y` for a new design matrix `X` using the fitted OLS model.
        If the model was fitted with intercept=True, then an intercept will
        be added prior to prediction.
        """
        if self.intercept is not None:
            return(
                sparse.hstack(X, np.ones(X.shape[0])).T *
                np.concatenate(self.beta, self.intercept, axis=None)
            )
        else:
            return X.T * self.beta


    def summary(self):
        """
        TODO: print a pretty summary
        """
        pass
