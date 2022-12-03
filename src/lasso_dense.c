#include <assert.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include "pcg_basic.h"


#define MAX_THREADS 128
#define INFO(msg) printf("lslasso: " msg "\n");


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
	const gsl_matrix* X,
	const gsl_vector* y,
	gsl_vector* gamma,
	gsl_vector* buffer)
{
	gsl_vector_set_zero(gamma);

	/* St = XX:*(v*J(1,cols(XX),1)) */
        /* Ups = sqrt(colsum((St):^2)/nObs) */

	for (size_t i=0; i<y->size; i++) {
		gsl_vector_const_view col = gsl_matrix_const_column(X, i);
		gsl_vector_memcpy(buffer, &col.vector);
		gsl_vector_scale(buffer, gsl_vector_get(y, i));
		gsl_vector_mul(buffer, buffer);
		gsl_vector_add(gamma, buffer);
	}

	gsl_vector_scale(gamma, 1.0 / y->size);

	for (size_t i=0; i<gamma->size; i++) {
		gsl_vector_set(gamma, i, sqrt(gsl_vector_get(gamma, i)));
	}
}


static void _select_beta(
	const gsl_matrix* X,
	const gsl_vector* y,
	const gsl_vector* beta,
	double zeroTol,
	gsl_matrix* pnBuffer,
	gsl_matrix* ppBuffer,
	gsl_vector* pBuffer,
	gsl_vector* betaPL,
	gsl_vector* v)
{
	size_t p = X->size1;
	size_t n = X->size2;
	size_t pSelect = 0;

	gsl_vector_set_zero(betaPL);
	gsl_vector_memcpy(v, y);

	for (size_t i=0; i<p; i++)
	{
		double b = gsl_vector_get(beta, i);
		if (fabs(b) > zeroTol)
		{
			gsl_vector_set(betaPL, pSelect, b);
			/* X is in col-major order, so we can copy rows */
			gsl_vector_view row1 = gsl_matrix_row(pnBuffer, pSelect);
			gsl_vector_const_view row2 = gsl_matrix_const_row(X, i);
			gsl_vector_memcpy(&row1.vector, &row2.vector);
			pSelect++;
		}
	}

	if (pSelect == 0)
	{
		INFO("beta is all zero");
		return;
	}

	/* setup views based on pSelect */

	gsl_matrix_view vX  = gsl_matrix_submatrix(pnBuffer, 0, 0, pSelect, n);
	gsl_matrix_view vXX = gsl_matrix_submatrix(ppBuffer, 0, 0, pSelect, pSelect);

	gsl_vector_view vXy = gsl_vector_subvector(pBuffer, 0, pSelect);
	gsl_vector_view vb  = gsl_vector_subvector(betaPL,  0, pSelect);

	/* Calculate X'X and X'y. */
	gsl_blas_dgemm(
		CblasNoTrans, CblasTrans, 1.0, &vX.matrix, &vX.matrix,
		0.0, &vXX.matrix);
	gsl_blas_dgemv(CblasNoTrans, 1.0, &vX.matrix, y, 0.0, &vXy.vector);

	/* Solve for betaPL. */
	_lusolve(&vXX.matrix, &vXy.vector, &vb.vector);

	/* v = y - X betaPL */
	gsl_blas_dgemv(CblasTrans, -1.0, &vX.matrix, &vb.vector, 1.0, v);
}


static void _run_shotgun(
	const gsl_matrix* XX,
	const gsl_vector* Xy, 
	const gsl_vector* XX_diag_inv,
	const gsl_vector* lambda_gamma,
	double shotgunTol,
	gsl_vector* buffer,
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


void sparselm_lasso_dense(
	const gsl_matrix* X,
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
	gsl_vector* pBuffer      = gsl_vector_calloc(p);
	gsl_vector* v            = gsl_vector_calloc(n);

	INFO("allocating matrices");

	gsl_matrix* XX       = gsl_matrix_calloc(p, p);
	gsl_matrix* ppBuffer = gsl_matrix_calloc(p, p);
	gsl_matrix* pnBuffer = gsl_matrix_calloc(p, n);

	/* Default formula for lambda. */
	if (lambda == 0.0) {
		INFO("setting lambda");
		// 2.2*sqrt(2*`nUse'*log(2*`p'/(.1/log(`nUse'))))
		lambda = 2.2*sqrt(2.*n*log(2.*p/(.1/log((double)n))));
	}

	INFO("calculating X'X and X'y");

	/* Calculate X'X and X'y. Note: since the input X is column-ordered, and GSL
         * uses row-ordering, we treat X as if it is X' and reverse the transpose flags. */
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, X, X, 0.0, XX);
	gsl_blas_dgemv(CblasNoTrans, 1.0, X, y, 0.0, Xy);

	/* Initial value of beta */
	if (gsl_blas_dasum(beta) == 0)
	{
		INFO("solving for initial beta");

		/* beta=lusolve(XX+lambda*I(p),Xy) */
		gsl_matrix_memcpy(ppBuffer, XX);

		for (size_t i=0; i<p; i++)
			gsl_matrix_set(ppBuffer, i, i, gsl_matrix_get(XX, i, i) + lambda);

		_lusolve(ppBuffer, Xy, beta);
	}
	else
	{
		INFO("warm starting with non-zero beta");
	}
	
	/* Scale X'X and X'y by 2.0 */
	gsl_matrix_scale(XX, 2.0);
	gsl_vector_scale(Xy, 2.0);

	/* Pre-calculate 1.0 / diag(X'X) */
	for (size_t i=0; i<p; i++)
		gsl_vector_set(XX_diag_inv, i, 1.0 / gsl_matrix_get(XX, i, i));

	INFO("setting initial gamma from y");

	_update_penalties(X, y, gamma, pBuffer);

	/* Initial shotgun uses 1/2 lamba */

	gsl_vector_memcpy(lambda_gamma, gamma);
	gsl_vector_scale(lambda_gamma, 0.5*lambda);

	/* Repeat shotgun until convergence. */
	double diff;
	do
	{
		INFO("shotgun iteration");

		_run_shotgun(
			XX, Xy, XX_diag_inv, lambda_gamma,
			shotgunTol, pBuffer, beta);

		INFO("selecting non-zero beta");

		_select_beta(
			X, y, beta,
			zeroTol,
			pnBuffer, ppBuffer, pBuffer,
			betaPL, v);

		INFO("updating gamma");

		gsl_vector_memcpy(gamma0, gamma);

		_update_penalties(X, v, gamma, pBuffer);

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
	gsl_vector_free(pBuffer);
	gsl_vector_free(v);

	INFO("freeing matrices");

	gsl_matrix_free(XX);
	gsl_matrix_free(pnBuffer);
	gsl_matrix_free(ppBuffer);
}
