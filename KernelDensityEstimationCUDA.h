#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "CudaErrorHandler.h"
#include <thrust/device_vector.h>

class Volume;
#include "KernelBandwidth.h";

class KernelDensityEstimationCUDA
{
public:
	KernelDensityEstimationCUDA();
	//void setSamplesFromHost(const float4* samples_host, int N);

	//void setVolume(std::shared_ptr<Volume>);
	//std::weak_ptr<Volume> getVolume() const;

	void estimateAccurate(Volume&, const float4* samples_device,int nsamples, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
	void estimateBinned(Volume& volume, const float4* samples_device, int nsamples, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
	void estimateBinned(Volume& binned_volume, const KernelBandwidth& kernelBandwidth = KernelBandwidth());
private:

};