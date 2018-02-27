#include "FFTKDECUDAMemoryManager.h"
#include "CudaErrorHandler.h"


FFTKDECUDAMemoryManager::FFTKDECUDAMemoryManager():
	allocated_element_num(0),
	Czp(nullptr),
	Kzp(nullptr),
	FCzp(nullptr),
	FKzp(nullptr)
{
}

void FFTKDECUDAMemoryManager::allocate(int3 P)
{
	std::size_t needed_elements = P.x*P.y*P.z;
	if (needed_elements <= allocated_element_num) return;
	release();
	chkCudaErrors(cudaMalloc(&Czp,needed_elements*sizeof(float)));
	chkCudaErrors(cudaMalloc(&Kzp, needed_elements * sizeof(float)));
	chkCudaErrors(cudaMalloc(&FCzp, needed_elements * sizeof(cuFloatComplex)));
	chkCudaErrors(cudaMalloc(&FKzp, needed_elements * sizeof(cuFloatComplex)));
	allocated_element_num = needed_elements;
}

void FFTKDECUDAMemoryManager::allocateFFTWorkArea(size_t size)
{
	if (fftWorkAreaPtr != nullptr)
	{
		chkCudaErrors(cudaFree(fftWorkAreaPtr));
		fftWorkAreaAllocated = 0;
	}
	if (size > fftWorkAreaAllocated)
	{
		chkCudaErrors(cudaMalloc(&fftWorkAreaPtr, size));
		fftWorkAreaAllocated = size;
	}

}

void FFTKDECUDAMemoryManager::release()
{
	if (Czp)
	{
		chkCudaErrors(cudaFree(Czp)); Czp = nullptr;
	}
	if (Kzp)
	{
		chkCudaErrors(cudaFree(Kzp)); Kzp = nullptr;
	}
	if (FCzp)
	{
		chkCudaErrors(cudaFree(FCzp)); FCzp = nullptr;
	}
	if (FKzp)
	{
		chkCudaErrors(cudaFree(FKzp)); FKzp = nullptr;
	}
}

FFTKDECUDAMemoryManager::~FFTKDECUDAMemoryManager()
{
	release();
	if (fftWorkAreaPtr != nullptr)
	{
		chkCudaErrors(cudaFree(fftWorkAreaPtr));
		fftWorkAreaAllocated = 0;
	}
}

std::shared_ptr<FFTKDECUDAMemoryManager> FFTKDECUDAMemoryManager::New()
{
	return std::make_shared<FFTKDECUDAMemoryManager>();
}