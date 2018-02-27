#pragma once
#ifndef GAUSSIANKERNEL_CUH
#define GAUSSIANKERNEL_CUH

#include "KernelBandwidth.h"

__constant__ SymmetricMatrix H_inv;

__constant__ float kernel_normalizing_constant;

inline __device__ float GaussianKernel(float4 pos, float4 sample)
{
	register float3 pms = (make_float3(pos) - make_float3(sample)) / sample.w;
	register float3 tmp = H_inv.mult(pms);

	return kernel_normalizing_constant / (sample.w*sample.w*sample.w)*exp(-0.5f*dot(pms, tmp));
}

inline __device__ float GaussianKernel(float3 pos, float3 sample)
{
	register float3 pms = pos - sample;
	register float3 tmp = H_inv.mult(pms);

	return kernel_normalizing_constant * exp(-0.5f*dot(pms, tmp));
}

#endif