#pragma once
#include <cuda_runtime.h>


// bounding box type
struct BoundingBox
{
	// construct an empty box
	__host__ __device__
		BoundingBox() {}

	// construct a box from a single point
	__host__ __device__
		BoundingBox(const float4 &point)
		: lower(point), upper(point)
	{}

	// construct a box from a pair of points
	__host__ __device__
		BoundingBox(const float4 &ll, const float4 &ur)
		: lower(ll), upper(ur)
	{}

	float4 lower, upper;
};

BoundingBox estimateBoundingBox(const float4* samples_device, int nsamples);