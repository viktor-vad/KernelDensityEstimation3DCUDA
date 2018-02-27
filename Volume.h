#pragma once

#include <cuda_runtime.h>
#include <helper_math.h>
#include "CudaErrorHandler.h"

#include "BoundingBox.h"

struct GPUVolumeData
{
	unsigned int width_ = 0, height_ = 0, depth_ = 0;
	BoundingBox BB_;
	float4 cellSize_;
	cudaPitchedPtr pitchedDevPtr = { 0 };

	//unsigned int slicePitch = 0;
	inline __device__ __forceinline__ unsigned int numberOfCells() const
	{
		return width_*height_*depth_;
	}
	inline __device__ __forceinline__ uint3 gridSize() const
	{
		return make_uint3(width_, height_, depth_);
	}
	__device__ float& operator()(const int i, const int j, const int k)
	{
		char* rowPtr = ((char*)pitchedDevPtr.ptr) + (pitchedDevPtr.ysize*k + j)*pitchedDevPtr.pitch;
		return ((float*)rowPtr)[i];
	}
	inline __device__ __forceinline__ float4 cellCenter(const int i, const int j, const int k) const
	{
		return make_float4(float(i) + 0.5f, float(j) + 0.5f, float(k) + 0.5f, 0.0)*cellSize_ + BB_.lower;
	}

};



class Volume
{

public:
	Volume();
	~Volume();
	void setGridSize(const uint3 size);
	void setGridSize(const unsigned int w, const int h, const int d);

	inline uint3 Volume::gridSize() const
	{
		return make_uint3(width_, height_, depth_);
	}

	void setBox(float4 minBB, float4 maxBB);
	inline void setBox(BoundingBox BB)
	{
		setBox(BB.lower, BB.upper);
	}
	void getBox(float4& minBB, float4& maxBB) const;
	inline BoundingBox getBox() const
	{
		return BB_;
	}
	inline float3 cellSize() const
	{
		return make_float3(cellSize_.x, cellSize_.y, cellSize_.z);
	}
	inline float4 cellCenter(const int i, const int j, const int k) const
	{
		return make_float4(float(i) + 0.5f, float(j) + 0.5f, float(k) + 0.5f, 0.0)*cellSize_ + BB_.lower;
	}
	inline unsigned int numberOfCells() const
	{
		return width_*height_*depth_;
	}

	float valueAt(const int i, const int j, const int k) const;

	void fillWithZeros();
	
	GPUVolumeData getVolumeData() const;

	void copyFlattened(float* dst, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) const;
private:
	unsigned int width_=0, height_=0, depth_=0;
	BoundingBox BB_;
	float4 cellSize_;
	cudaPitchedPtr pitchedDevPtr = {0};

	unsigned int slicePitch=0;
};