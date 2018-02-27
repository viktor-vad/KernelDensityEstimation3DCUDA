#pragma once

#include "FFTKDEEstimationBaseCUDA.h"
#include "KernelBandwidth.h"

class FFTLSCVEstimatorCUDA :public FFTKDEEstimationBaseCUDA
{
public:
	FFTLSCVEstimatorCUDA();
	//void setSamples(const float4* samples_device, int nsamples,uint3 bin_num,const float4* bb_hint=nullptr);
	
	float estimateLSCV(const Volume& binned_volume, int nsamples, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
private:

	//Volume volume;
	//int nsamples;
};