#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <cufft.h>

#include <helper_cuda.h>

#ifndef KDE_CUDA_CUSTOM_ERROR_HANDLER
#define chkCudaErrors(val)         checkCudaErrors(val)
#else
#include <kdeCudaCustomErrorHandler.h>

template< typename T >
inline void kde_check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		char msg[1024];
		sprintf(msg, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		KDE_CUDA_ERROR_HANDLER(msg);
	}
}

#define chkCudaErrors(val)  kde_check ( (val), #val, __FILE__, __LINE__ )

#endif
