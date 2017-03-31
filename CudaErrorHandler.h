#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <cufft.h>

#include <helper_cuda.h>

#define chkCudaErrors(val)          CudaErrorHandler::check_cuda ( (val), #val, __FILE__, __LINE__ )
//
//struct KDE_CUDA_ERROR_HANDLER_
//{
//	std::function<void(cudaError_t result, char const *const func, const char *const file, int const line, const char* message)> KDE_CUDA_ERROR_HANDLER_FUNC;
//};
//
//extern KDE_CUDA_ERROR_HANDLER_ KDE_CUDA_ERROR_HANDLER;
//
//void setKdeCUDAErrorHandler(std::function<void(cudaError_t result, char const *const func, const char *const file, int const line, const char* message)>);
//
//inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
//{
//	if (result)
//	{
//		const char* msg = cudaGetErrorString(result);
//		KDE_CUDA_ERROR_HANDLER.KDE_CUDA_ERROR_HANDLER_FUNC(result, func, file, line, msg);
//	}
//}
//
//

struct CudaErrorHandler
{

	static std::function<void(char const *const func, const char *const file, int const line, const char* message)> KDE_CUDA_ERROR_HANDLER_FUNC;
//protected:
	static inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
	{
		if (result)
		{
			const char* msg = cudaGetErrorString(result);
			KDE_CUDA_ERROR_HANDLER_FUNC(func, file, line, msg);
		}
	}
#ifdef CUFFTAPI
	static inline void check_cuda(cufftResult_t result, char const *const func, const char *const file, int const line)
	{
		if (result)
		{
			const char* msg = _cudaGetErrorEnum(result);
			KDE_CUDA_ERROR_HANDLER_FUNC( func, file, line, msg);
		}
	}
#endif
	//CudaErrorHandler();
};



