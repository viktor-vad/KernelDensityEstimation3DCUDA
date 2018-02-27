#include "FFTKDEEstimationBaseCUDA.h"

#include "FFTKDECUDAMemoryManager.h"

FFTKDEEstimationBaseCUDA::FFTKDEEstimationBaseCUDA() :
	memoryManager(std::make_shared<FFTKDECUDAMemoryManager>())
{
}

void FFTKDEEstimationBaseCUDA::setMemoryManager(std::shared_ptr<FFTKDECUDAMemoryManager> otherMemoryManager)
{
	memoryManager = otherMemoryManager;
}
