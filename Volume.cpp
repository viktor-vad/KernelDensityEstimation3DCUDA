#include "Volume.h"
#include <helper_cuda.h>

Volume::Volume()
{

}

Volume::~Volume()
{
	if (pitchedDevPtr.ptr)
		cudaFree(pitchedDevPtr.ptr);
}

void Volume::setGridSize(const unsigned int w, const int h, const int d)
{
	width_ = w;
	height_ = h;
	depth_ = d;
	cellSize_ = (BB_.upper - BB_.lower) / make_float4(w, h, d, 1.0);
	if (pitchedDevPtr.ptr)
		cudaFree(pitchedDevPtr.ptr);

	register cudaExtent mem_extent = make_cudaExtent(w * sizeof(float), h, d);
	chkCudaErrors(cudaMalloc3D(&pitchedDevPtr, mem_extent));
	//slicePitch = pitchedDevPtr.pitch*pitchedDevPtr.ysize;
}

void Volume::setGridSize(const uint3 size)
{
	setGridSize(size.x, size.y, size.z);
}


void  Volume::setBox(float4 minBB, float4 maxBB)
{
	BB_.lower = minBB;
	BB_.upper = maxBB;
	cellSize_ = (BB_.upper - BB_.lower) / make_float4(width_, height_, depth_, 1.0);
}

void Volume::getBox(float4& minBB, float4& maxBB) const
{
	minBB = BB_.lower;
	maxBB = BB_.upper;
}

void  Volume::fillWithZeros()
{
	chkCudaErrors(cudaMemset3D(pitchedDevPtr, 0, make_cudaExtent(width_ * sizeof(float), height_, depth_)));
}

GPUVolumeData Volume::getVolumeData() const
{
	GPUVolumeData volData;
	volData.width_ = width_;
	volData.height_ = height_;
	volData.depth_ = depth_;
	volData.cellSize_ = cellSize_;
	volData.BB_ = BB_;
	volData.pitchedDevPtr = pitchedDevPtr;
	//volData.slicePitch = slicePitch;

	return volData;
}

void Volume::copyFlattened(float* dst, cudaMemcpyKind kind) const
{
	cudaMemcpy3DParms myParms = { 0 };
	myParms.srcPtr = pitchedDevPtr;
	myParms.dstPtr = make_cudaPitchedPtr(dst, width_ * sizeof(float), width_ * sizeof(float), height_);
	myParms.extent = make_cudaExtent(width_ * sizeof(float), height_, depth_);
	myParms.kind = kind;

	chkCudaErrors(cudaMemcpy3D(&myParms));
}

float Volume::valueAt(const int i, const int j, const int k) const
{
	float value;

	cudaMemcpy3DParms myParms = { 0 };
	myParms.srcPtr = pitchedDevPtr;
	myParms.dstPtr = make_cudaPitchedPtr(&value, sizeof(float), sizeof(float), 1);
	myParms.extent = make_cudaExtent(sizeof(float), 1, 1);
	myParms.kind = cudaMemcpyDeviceToHost;
	myParms.srcPos = make_cudaPos(i * sizeof(float), j, k);
	chkCudaErrors(cudaMemcpy3D(&myParms));

	return value;
}