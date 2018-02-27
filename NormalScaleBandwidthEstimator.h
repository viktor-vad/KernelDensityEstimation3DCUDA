#pragma once
#include <cuda_runtime.h>

#include <memory>

class KernelBandwidth;

class NormalScaleBandwidthEstimator
{
public:

	static void estimateBandwidth(const float4* samples_device, const int sample_num, KernelBandwidth&);
	static void estimateBandwidthAA(const float4* samples_device, const int sample_num, KernelBandwidth&);
};

