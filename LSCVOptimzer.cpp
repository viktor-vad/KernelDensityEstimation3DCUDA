#include "LSCVOptimzer.h"

#include <KernelBandwidth.h>
#include <Volume.h>
#include <FFTLSCVEstimatorCUDA.h>

LSCVOptimizer::LSCVOptimizer(Volume* pVolume, int samples_num) :
	pVolume_(pVolume)
	, samples_num_(samples_num)
	,lscvEstimator(std::make_unique<FFTLSCVEstimatorCUDA>())
{
	lbfgs_parameter_init(&param);
	//param.max_step = 1.0f;
	//param.min_step = 1.0f;
	spatial_unit_step = fminf(pVolume_->cellSize().x,fminf(pVolume_->cellSize().y, pVolume_->cellSize().z));
}

int LSCVOptimizer::Optimize(KernelBandwidth& kernelBandwidth)
{
	kernelBandwidth.getCoeffs(vechH_[0], vechH_[1], vechH_[2], vechH_[3], vechH_[4], vechH_[5]);

	lbfgsfloatval_t fx;
	int ret = lbfgs(6, vechH_, &fx, lbfgs_evaluate_, lbfgs_progress_, this, &param);
	kernelBandwidth.setBandwidthMatrix(vechH_[0], vechH_[1], vechH_[2], vechH_[3], vechH_[4], vechH_[5]);
	checkEigenValues(kernelBandwidth);
	return ret;
}

void LSCVOptimizer::checkEigenValues(KernelBandwidth& kernelBandwidth_)
{
	if (kernelBandwidth_.getSmallestEigenValue() < spatial_unit_step*spatial_unit_step)
	{
		float eig1, eig2, eig3;
		float3 eigv1, eigv2, eigv3;
		kernelBandwidth_.getEigenSystem(eig1, eig2, eig3, eigv1, eigv2,eigv3);
		kernelBandwidth_.setFromEigenSystem(fmaxf(fabsf(eig1), 4.0f*spatial_unit_step*spatial_unit_step), fmaxf(fabs(eig2), 4.0f*spatial_unit_step*spatial_unit_step), fmaxf(fabsf(eig3), 4.0f*spatial_unit_step*spatial_unit_step), eigv1, eigv2, eigv3);
	}
}

lbfgsfloatval_t LSCVOptimizer::evaluate_(const lbfgsfloatval_t *vechH, lbfgsfloatval_t *g, const lbfgsfloatval_t step)
{
	kernelBandwidth_.setBandwidthMatrix(vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
	
	checkEigenValues(kernelBandwidth_);

	float lscv = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);

	kernelBandwidth_.setBandwidthMatrix(vechH[0] + spatial_unit_step, vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
	checkEigenValues(kernelBandwidth_);
	register float tmp = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);
	g[0] = (tmp - lscv) / spatial_unit_step;
	if (tmp != tmp) g[0] = 0.0f;

	kernelBandwidth_.setBandwidthMatrix(vechH[0], vechH[1] + spatial_unit_step, vechH[2], vechH[3], vechH[4], vechH[5]);
	checkEigenValues(kernelBandwidth_);
	tmp = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);
	g[1] = (tmp - lscv) / spatial_unit_step;
	if (tmp != tmp) g[1] = 0.0f;

	kernelBandwidth_.setBandwidthMatrix(vechH[0], vechH[1], vechH[2] + spatial_unit_step, vechH[3], vechH[4], vechH[5]);
	checkEigenValues(kernelBandwidth_);
	tmp = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);
	g[2] = (tmp - lscv) / spatial_unit_step;
	if (tmp != tmp) g[2] = 0.0f;

	kernelBandwidth_.setBandwidthMatrix(vechH[0], vechH[1], vechH[2], vechH[3] + spatial_unit_step, vechH[4], vechH[5]);
	checkEigenValues(kernelBandwidth_);
	tmp = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);
	g[3] = (tmp - lscv) / spatial_unit_step;
	if (tmp != tmp) g[3] = 0.0f;

	kernelBandwidth_.setBandwidthMatrix(vechH[0], vechH[1], vechH[2], vechH[3], vechH[4] + spatial_unit_step, vechH[5]);
	checkEigenValues(kernelBandwidth_);
	tmp = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);
	g[4] = (tmp - lscv) / spatial_unit_step;
	if (tmp != tmp) g[4] = 0.0f;

	kernelBandwidth_.setBandwidthMatrix(vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5] + spatial_unit_step);
	checkEigenValues(kernelBandwidth_);
	tmp = lscvEstimator->estimateLSCV(*pVolume_, samples_num_, kernelBandwidth_);
	g[5] = (tmp - lscv) / spatial_unit_step;
	if (tmp != tmp) g[5] = 0.0f;

	return lscv;
}

lbfgsfloatval_t LSCVOptimizer::lbfgs_evaluate_(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step)
{
	return reinterpret_cast<LSCVOptimizer*>(instance)->evaluate_(x, g, step);
}

int LSCVOptimizer::progress_(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int k, int ls)
{
	printf("fx %.15g xnorm %.15g gnorm %.15g step %.15g k %d ls %d\n", fx, xnorm, gnorm, step, k, ls);
	return (step<=1.0f);
}

int LSCVOptimizer::lbfgs_progress_(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls)
{
	return reinterpret_cast<LSCVOptimizer*>(instance)->progress_(x, g, fx, xnorm, gnorm, step, k, ls);
}