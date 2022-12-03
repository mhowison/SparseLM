#include <assert.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include "pcg_basic.h"


#define MAX_THREADS 128
#define INFO(msg) fprintf(stderr, "lslasso: " msg "\n");
#define INFO1(msg,val) fprintf(stderr, "lslasso: " msg "%zu\n", val);


static void _sp2d(
	gsl_matrix* dst,
	const gsl_spmatrix* src)
{
	assert(dst->size1 == src->size1);
	assert(dst->size2 == src->size2);
	for (size_t j=0; j<dst->size2; j++)
	{
		for (size_t i=0; i<dst->size1; i++)
		{
			gsl_matrix_set(dst, i, j, gsl_spmatrix_get(src, i, j));
		}
	}
}


static void _lusolve(
	gsl_matrix* A,
	const gsl_vector* b,
	gsl_vector* x)
{
	int s;

	assert(A->size1 == A->size2);
	assert(A->size1 == b->size);
	assert(A->size1 == x->size);

	gsl_permutation* p = gsl_permutation_calloc(A->size1);

	gsl_linalg_LU_decomp(A, p, &s);
	gsl_linalg_LU_solve(A, p, b, x);

	gsl_permutation_free(p);
}


static void _update_penalties(
	const gsl_spmatrix* X,
	const gsl_vector* y,
	gsl_vector* gamma)
{
	assert(GSL_SPMATRIX_ISTRIPLET(X));
	assert(X->size1 == gamma->size);
	assert(X->size2 == y->size);

	gsl_vector_set_zero(gamma);

	/* St = XX:*(v*J(1,cols(XX),1)) */
        /* Ups = sqrt(colsum((St):^2)/nObs) */

        for (size_t k=0; k<X->nz; k++)
	{
		size_t i = X->i[k];
		size_t j = X->p[k];
		double x = X->data[k] * gsl_vector_get(y, j);
		gsl_vector_set(gamma, i, gsl_vector_get(gamma, i) + x*x);
	}

	gsl_vector_scale(gamma, 1.0 / y->size);

	for (size_t i=0; i<gamma->size; i++)
	{
		gsl_vector_set(gamma, i, sqrt(gsl_vector_get(gamma, i)));
	}
}


static void _select_beta(
	const gsl_spmatrix* X,
	const gsl_vector* y,
	const gsl_vector* beta,
	double zeroTol,
	gsl_vector* betaPL,
	gsl_vector* v)
{
	size_t p = X->size1;
	size_t n = X->size2;

	assert(y->size == n);
	assert(beta->size == p);
	assert(betaPL->size == p);
	assert(v->size == n);

	gsl_vector_set_zero(betaPL);
	gsl_vector_memcpy(v, y);

	size_t ps = 0;
	gsl_vector_uint* index = gsl_vector_uint_alloc(p);

	/* Find non-zero betas */
	for (size_t i=0; i<p; i++)
	{
		if (fabs(gsl_vector_get(beta, i) > zeroTol))
		{
			gsl_vector_uint_set(index, ps++, i);
		}
	}

	if (ps == 0)
	{
		INFO("beta is all zero");
		return;
	}

	/* Copy selected columns from X */
	gsl_spmatrix* Xs = gsl_spmatrix_alloc_nzmax(ps, n, 1024*1024, GSL_SPMATRIX_TRIPLET);
	for (size_t i=0; i<ps; i++)
	{
		size_t ii = gsl_vector_uint_get(index, i);

		for (size_t j=0; j<n; j++)
		{
			double x = gsl_spmatrix_get(X, ii, j);
			if (x != 0.0) gsl_spmatrix_set(Xs, i, j, x);
		}
	}

	gsl_spmatrix* Xc = gsl_spmatrix_ccs(Xs);
	gsl_spmatrix_free(Xs);

	gsl_vector* Xy = gsl_vector_alloc(ps);
	gsl_spmatrix* XX = gsl_spmatrix_alloc_nzmax(ps, ps, ps*ps, GSL_SPMATRIX_CCS);

	size_t nnz = gsl_spmatrix_nnz(Xc);
	gsl_spmatrix* XcT = gsl_spmatrix_alloc_nzmax(n, ps, nnz, GSL_SPMATRIX_CCS);
	gsl_spmatrix_transpose_memcpy(XcT, Xc);

	/* Calculate selected X'X and X'y. */
	gsl_spblas_dgemm(1.0, Xc, XcT, XX);
	gsl_spblas_dgemv(CblasNoTrans, 1.0, Xc, y, 0.0, Xy);

	gsl_spmatrix_free(XcT);

	/* Convert XXs to dense matrix. */
	gsl_matrix* XX_dense = gsl_matrix_alloc(ps, ps);
	_sp2d(XX_dense, XX);
	gsl_spmatrix_free(XX);

	gsl_vector_view b = gsl_vector_subvector(betaPL, 0, ps);

	/* Solve for betaPL. */
	_lusolve(XX_dense, Xy, &b.vector);

	/* v = y - X betaPL */
	gsl_spblas_dgemv(CblasTrans, -1.0, Xc, &b.vector, 1.0, v);

	gsl_vector_uint_free(index);
	gsl_spmatrix_free(Xc);
	gsl_vector_free(Xy);
	gsl_matrix_free(XX_dense);
}


static void _run_shotgun(
	const gsl_matrix* XX,
	const gsl_vector* Xy, 
	const gsl_vector* XX_diag_inv,
	const gsl_vector* lambda_gamma,
	double shotgunTol,
	gsl_vector* beta)
{
	int nt = omp_get_num_threads();
	assert (nt <= MAX_THREADS);
	double delta[MAX_THREADS];

	do
	{
		for (size_t i=0; i<nt; i++) delta[i] = 0.0;

		#pragma omp parallel for
		for (size_t i=0; i<beta->size; i++)
		{
			/* choose a coordinate from the randomly shuffled list */
			size_t j = i;
			double beta_j = gsl_vector_get(beta, j);
			if (beta_j == 0.0) continue; 

			/* shoot */
			double S0 = -gsl_matrix_get(XX, j, j) * beta_j - gsl_vector_get(Xy, j);
			for (size_t k=0; k<beta->size; k++)
			{
				double beta_k = gsl_vector_get(beta, k);
				if (beta_k != 0.0) S0 += gsl_matrix_get(XX, j, k) * beta_k;
			}

			double l = gsl_vector_get(lambda_gamma, j);
			if (S0 > l)
			{
				S0 = (l - S0) * gsl_vector_get(XX_diag_inv, j);
			}
			else if (S0 < -l)
			{
				S0 = (-l - S0) * gsl_vector_get(XX_diag_inv, j);
			}
			else
			{
				S0 = 0.0;
			}

			/*double maxdelta0 = maxdelta;
			while(!__atomic_compare_exchange_n(
				&maxdelta,
				&maxdelta0,
				fmax(maxdelta0, fabs(beta_j - S0)),
				0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { maxdelta0 = maxdelta; }*/
			int t = omp_get_thread_num();
			delta[t] = fmax(delta[t], fabs(beta_j - S0));

			gsl_vector_set(beta, j, S0);
		}

		for (size_t i=1; i<nt; i++) delta[0] = fmax(delta[0], delta[i]);

		/*#pragma omp master
		{
			for (size_t i=1; i<nt; i++) diff[0] = fmax(diff[0], diff[i]);
			//gsl_vector_sub(beta0, beta);
			//diff[0] = gsl_blas_dasum(beta0);
			//gsl_vector_memcpy(beta0, beta);
			printf("%f\n", diff[0]);
		}
		printf("%d at barrier\n", t);
		#pragma omp barrier
		printf("%d past barrier\n", t);*/
		printf("%f\n", delta[0]);
	}
	while (delta[0] > shotgunTol);
}


void sparselm_lasso_sparse(
	const gsl_spmatrix* X,
	const gsl_vector* y,
	double lambda,
	double gammaTol,
	double shotgunTol,
	double zeroTol,
	gsl_vector* beta,
	gsl_vector* betaPL)
{
	/* dimensions */
	size_t p = X->size1;
	size_t n = X->size2;
	INFO1("p = ", p);
	INFO1("n = ", n);

	assert(y->size == n);
	assert(beta->size == p);
	assert(betaPL->size == p);

	INFO("allocating vectors");

	gsl_vector* Xy           = gsl_vector_calloc(p);
	gsl_vector* XX_diag_inv  = gsl_vector_calloc(p);
	gsl_vector* lambda_gamma = gsl_vector_calloc(p);
	gsl_vector* gamma        = gsl_vector_calloc(p);
	gsl_vector* gamma0       = gsl_vector_calloc(p);
	gsl_vector* beta0        = gsl_vector_calloc(p);
	gsl_vector* v            = gsl_vector_calloc(n);

	INFO("allocating matrices");

	/* Convert sparse triplet format to CCS */
	gsl_spmatrix* Xc  = gsl_spmatrix_ccs(X);
	gsl_spmatrix* XXc = gsl_spmatrix_alloc_nzmax(p, p, p*p, GSL_SPMATRIX_CCS);
	gsl_matrix*   XX  = gsl_matrix_alloc(p, p);

	/* spblas requires an explicit transpose */
	size_t nnz = gsl_spmatrix_nnz(Xc);
	gsl_spmatrix* XcT = gsl_spmatrix_alloc_nzmax(n, p, nnz, GSL_SPMATRIX_CCS);
	gsl_spmatrix_transpose_memcpy(XcT, Xc);

	/* Default formula for lambda. */
	if (lambda == 0.0) {
		INFO("setting lambda");
		// 2.2*sqrt(2*`nUse'*log(2*`p'/(.1/log(`nUse'))))
		lambda = 2.2*sqrt(2.*n*log(2.*p/(.1/log((double)n))));
	}

	INFO("calculating X'X and X'y");

	/* Calculate X'X and X'y. Note: since the input X is column-ordered, and GSL
         * uses row-ordering, we treat X as if it is X' and reverse the transpose flags. */
	gsl_spblas_dgemm(1.0, Xc, XcT, XXc);
	gsl_spblas_dgemv(CblasNoTrans, 1.0, Xc, y, 0.0, Xy);

	gsl_spmatrix_free(Xc);
	gsl_spmatrix_free(XcT);

	/* Initial value of beta */
	if (gsl_blas_dasum(beta) == 0)
	{
		INFO("solving for initial beta");

		/* Convert XX to dense matrix. */
		_sp2d(XX, XXc);

		/* beta=lusolve(XX+lambda*I(p),Xy) */
		for (size_t i=0; i<p; i++)
			gsl_matrix_set(XX, i, i, gsl_matrix_get(XX, i, i) + lambda);

		_lusolve(XX, Xy, beta);

	}
	else
	{
		INFO("warm starting with non-zero beta");
	}

	_sp2d(XX, XXc);
	gsl_spmatrix_free(XXc);
	
	/* Scale X'X and X'y by 2.0 */
	gsl_matrix_scale(XX, 2.0);
	gsl_vector_scale(Xy, 2.0);

	/* Pre-calculate 1.0 / diag(X'X) */
	for (size_t i=0; i<p; i++)
		gsl_vector_set(XX_diag_inv, i, 1.0 / gsl_matrix_get(XX, i, i));

	INFO("setting initial gamma from y");

	_update_penalties(X, y, gamma);

	/* Initial shotgun uses 1/2 lamba */

	gsl_vector_memcpy(lambda_gamma, gamma);
	gsl_vector_scale(lambda_gamma, 0.5*lambda);

	/* Repeat shotgun until convergence. */
	double diff;
	do
	{
		INFO("shotgun iteration");

		_run_shotgun(XX, Xy, XX_diag_inv, lambda_gamma, shotgunTol, beta);

		INFO("selecting non-zero beta");

		_select_beta(X, y, beta, zeroTol, betaPL, v);

		INFO("updating gamma");

		gsl_vector_memcpy(gamma0, gamma);

		_update_penalties(X, v, gamma);

		if (gsl_blas_dasum(gamma) == 0)
		{
			INFO("gamma is all zero");
		}

		gsl_vector_memcpy(lambda_gamma, gamma);
		gsl_vector_scale(lambda_gamma, lambda);

		/* l2 norm of difference between new and old gamma */
		gsl_vector_sub(gamma0, gamma);
		diff = gsl_blas_dnrm2(gamma0);
		printf("gamma diff: %f\n", diff);
	}
	while (diff > gammaTol);

	INFO("freeing vectors");

	gsl_vector_free(Xy);
	gsl_vector_free(XX_diag_inv);
	gsl_vector_free(lambda_gamma);
	gsl_vector_free(gamma);
	gsl_vector_free(gamma0);
	gsl_vector_free(beta0);
	gsl_vector_free(v);

	INFO("freeing matrices");

	gsl_matrix_free(XX);
}
