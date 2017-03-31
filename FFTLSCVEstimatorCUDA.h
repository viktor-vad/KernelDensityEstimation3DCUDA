#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "CudaErrorHandler.h"
#include <thrust/device_vector.h>

#include "Volume.h"
#include "KernelBandwidth.h"

class FFTLSCVEstimatorCUDA 
{
public:
	//void setSamples(const float4* samples_device, int nsamples,uint3 bin_num,const float4* bb_hint=nullptr);
	static float estimateLSCV(Volume& binned_volume, int nsamples, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
private:

	//Volume volume;
	//int nsamples;
};