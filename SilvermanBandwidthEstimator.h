#pragma once
#include <cuda_runtime.h>

#include <memory>

class KernelBandwidth;

class SilvermanBandwidthEstimator
{
public:

	static void estimateBandwidth(const float4* samples_device, const int sample_num, KernelBandwidth&);
};

