#pragma once
#include <cstddef>
#include <cuComplex.h>
#include <memory>

class FFTKDECUDAMemoryManager
{
public:
	FFTKDECUDAMemoryManager();
	~FFTKDECUDAMemoryManager();
	void allocate(int3 P);
	void allocateFFTWorkArea(size_t size);
	void release();
	inline float* Czp_data()
	{
		return Czp;
	}
	inline float* Kzp_data()
	{
		return Kzp;
	}
	inline cuFloatComplex* FCzp_data()
	{
		return FCzp;
	}
	inline cuFloatComplex* FKzp_data()
	{
		return FKzp;
	}
	inline void* fftWorkArea()
	{
		return fftWorkAreaPtr;
	}
	static std::shared_ptr<FFTKDECUDAMemoryManager> New();
private:
	std::size_t allocated_element_num;
	float* Czp;
	float* Kzp;
	cuFloatComplex* FCzp;
	cuFloatComplex* FKzp;
	void* fftWorkAreaPtr = nullptr;
	size_t fftWorkAreaAllocated = 0;
};

