#include "AccurateLSCVEstimatorCUDA.h"

#include "GaussianKernel.cuh"

#include <thrust/device_vector.h>

//
//__global__ void computeLSCVSummands(const float4* samples_device, int nsamples, float* lscv_summands)
//{
//	int idx = blockIdx.x*blockDim.x + threadIdx.x;
//	if (idx >= nsamples) return;
//	float3 pos_i = make_float3(samples_device[idx]);
//	register float res = 0.0f;
//	for (int i = 0; i < nsamples; ++i)
//	{
//		float3 pos = make_float3(samples_device[i]);
//		res+=GaussianKernel(make_float4(pos, 0.0), make_float4(pos_i, 2.0f)) - 2.0f*GaussianKernel(pos, pos_i);
//	}
//	lscv_summands[idx] = res;
//}


__global__ void computeLSCVSummands(const float4* samples_device, int nsamples, float* lscv_summands)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	register bool active = (idx < nsamples);
	register float3 pos_i;
	if (active) pos_i = make_float3(samples_device[idx]);
	else pos_i = make_float3(3.402823e+38, 3.402823e+38, 3.402823e+38);

	register float res = 0.0f;
	extern __shared__ float3 shared_pos[];

	for (int offset = 0; offset < nsamples; offset += blockDim.x)
	{
		int offset_leap = min(blockDim.x, nsamples - offset);
		
		if (threadIdx.x<offset_leap) shared_pos[threadIdx.x] = make_float3(samples_device[offset+threadIdx.x]);
		__syncthreads();
		//if (active)
		{
			for (int i = 0; i < offset_leap; ++i)
			{
				float3 pos = shared_pos[i];
				res += GaussianKernel(make_float4(pos, 0.0), make_float4(pos_i, 2.0f)) - 2.0f*GaussianKernel(pos, pos_i);
			}
		}
	}
	if (active) lscv_summands[idx] = res;
}

float AccurateLSCVEstimatorCUDA::estimateLSCV(const float4* samples_device, int nsamples, const KernelBandwidth& kernelBandwidth)
{
	SymmetricMatrix H_inv_ = kernelBandwidth.getInverseMatrix();

	chkCudaErrors(cudaMemcpyToSymbol(H_inv, &H_inv_, sizeof(SymmetricMatrix)));

	float kernel_normalizing_constant_ = 0.0634936359342409 //1/(2 pi)^(3/2)
		/ (sqrtf(kernelBandwidth.getDeterminant()));
	chkCudaErrors(cudaMemcpyToSymbol(kernel_normalizing_constant, &kernel_normalizing_constant_, sizeof(float)));

	thrust::device_vector<float> lscv_summands(nsamples);
	int launch_block_size = 512;
	int launch_grid_size = (nsamples + launch_block_size - 1) / launch_block_size;

	computeLSCVSummands<<<launch_grid_size,launch_block_size, launch_block_size *sizeof(float3)>>>(samples_device,nsamples,thrust::raw_pointer_cast(&lscv_summands[0]));

	float temp = thrust::reduce(lscv_summands.begin(), lscv_summands.end(), 0.0f, thrust::plus<float>());

	return (temp / float(nsamples) + 2.0f*kernel_normalizing_constant_) / float(nsamples);
}