#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "CudaErrorHandler.h"
#include <thrust/device_vector.h>

#include "Volume.h"
#include "KernelBandwidth.h"

class AccurateLSCVEstimatorCUDA
{
public:
	static float estimateLSCV(const float4* samples_device, int nsamples, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
	//static float estimateLSCV(Volume& binned_volume, int nsamples, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
//private:

	//Volume volume;
	//int nsamples;
};
