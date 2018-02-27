#pragma once
#include <memory>

#include <lbfgs.h>

class Volume;
class KernelBandwidth;
#include "FFTLSCVEstimatorCUDA.h"

class LSCVOptimizer
{
public:
	LSCVOptimizer(Volume* pVolume, int samples_num);

	int Optimize(KernelBandwidth& kernelBandwidth);
	inline FFTLSCVEstimatorCUDA* getEstimator()
	{
		return lscvEstimator.get();
	}
private:
	int samples_num_;
	Volume* pVolume_;

	lbfgsfloatval_t evaluate_(const lbfgsfloatval_t *vechH, lbfgsfloatval_t *g, const lbfgsfloatval_t step);

	static lbfgsfloatval_t lbfgs_evaluate_(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step);

	int progress_(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int k, int ls);

	static int lbfgs_progress_(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls);
	std::unique_ptr<FFTLSCVEstimatorCUDA> lscvEstimator;
	lbfgsfloatval_t vechH_[6];

	lbfgs_parameter_t param;
	float spatial_unit_step;
	KernelBandwidth kernelBandwidth_;
	void checkEigenValues(KernelBandwidth& kernelBandwidth_);
};