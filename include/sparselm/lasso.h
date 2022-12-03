#ifndef __SPARSELM_LASSO_H__
#define __SPARSELM_LASSO_H__

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void sparselm_lasso_dense(
	const gsl_matrix* X,
	const gsl_vector* y,
	double lambda,
	double gammaTol,
	double shotgunTol,
	double zeroTol,M
	gsl_vector* beta,
	gsl_vector* betaPL);

void sparselm_lasso_sparse(
	const gsl_spmatrix* X,
	const gsl_vector* y,
	double lambda,
	double gammaTol,
	double shotgunTol,
	double zeroTol,
	gsl_vector* beta,
	gsl_vector* betaPL);

#endif
