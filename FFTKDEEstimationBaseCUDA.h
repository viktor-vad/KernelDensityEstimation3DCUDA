#pragma once
#include <memory>
#include <cuda_runtime.h>

class FFTKDECUDAMemoryManager;
class GPUVolumeData;
class KernelBandwidth;

class FFTKDEEstimationBaseCUDA
{
public:
	void setMemoryManager(std::shared_ptr<FFTKDECUDAMemoryManager> otherMemoryManager);
protected:
	FFTKDEEstimationBaseCUDA();
	
	template <class KernelFiller>
	void estimateBinnedBase(GPUVolumeData& volData,int3& L,int3& P, const KernelBandwidth& kernelBandwidth);
	std::shared_ptr<FFTKDECUDAMemoryManager> memoryManager;
	
	float kernel_normalizing_constant_;
};