#pragma once

#include "Volume.h"
#include "VolumeBinning.h"

template <typename CONST_SAMPLES_IT_TYPE>
__global__ void buildBins3d(CONST_SAMPLES_IT_TYPE  samples, GPUVolumeData volume, const int nsamples)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= nsamples) return;
	float3 grid_pos = (make_float3(samples[idx]) - make_float3(volume.BB_.lower)) / make_float3(volume.cellSize_) - 0.5f;
	float3 grid_idx_f = floorf(grid_pos);
	int3 grid_idx = make_int3(grid_idx_f.x, grid_idx_f.y, grid_idx_f.z);
	grid_idx -= make_int3(grid_idx.x == volume.width_, grid_idx.y == volume.height_, grid_idx.z == volume.depth_);

	atomicAdd(&volume(grid_idx.x, grid_idx.y, grid_idx.z), 1.0f);
}

template <typename CONST_SAMPLES_IT_TYPE>
__global__ void buildLinearBins3d(CONST_SAMPLES_IT_TYPE  samples, GPUVolumeData volume, const int nsamples)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= nsamples) return;
	float3 cell_pos = (make_float3(samples[idx]) - make_float3(volume.BB_.lower)) / make_float3(volume.cellSize_);
	
	if (cell_pos.x<0.0f || cell_pos.x >volume.width_ ||
		cell_pos.y<0.0f || cell_pos.y >volume.height_ ||
		cell_pos.y<0.0f || cell_pos.z >volume.depth_) return;

	cell_pos -= 0.5f;

	float3 cell_idx_f = floorf(cell_pos);
	int3 cell_idx = make_int3(cell_idx_f.x, cell_idx_f.y, cell_idx_f.z);

	float3 cell_alpha = cell_pos - cell_idx_f;

	float3 cell_self_weight = make_float3(1.0f) - cell_alpha;
	float3 cell_neigh_weight = cell_alpha;

	if (cell_idx.x == -1)
	{
		cell_self_weight.x = cell_neigh_weight.x;
		cell_neigh_weight.x = 0.0f;
		cell_idx.x = 0;
	}

	if (cell_idx.y == -1)
	{
		cell_self_weight.y = cell_neigh_weight.y;
		cell_neigh_weight.y = 0.0f;
		cell_idx.y = 0;
	}

	if (cell_idx.z == -1)
	{
		cell_self_weight.z = cell_neigh_weight.z;
		cell_neigh_weight.z = 0.0f;
		cell_idx.z = 0;
	}
	if (cell_idx.x == volume.width_-1)
	{
		cell_neigh_weight.x = cell_self_weight.x;
		cell_self_weight.x = 0.0f;
		cell_idx.x--;
	}
	if (cell_idx.y == volume.height_ - 1)
	{
		cell_neigh_weight.y = cell_self_weight.y;
		cell_self_weight.y = 0.0f;
		cell_idx.y--;
	}
	if (cell_idx.z == volume.depth_ - 1)
	{
		cell_neigh_weight.z = cell_self_weight.z;
		cell_self_weight.z = 0.0f;
		cell_idx.z--;
	}

	//0,0,0
	register float temp = (cell_self_weight.x)*(cell_self_weight.y)*(cell_self_weight.z);
	atomicAdd(&volume(cell_idx.x, cell_idx.y, cell_idx.z), temp);
	//1,0,0
	temp = (cell_neigh_weight.x)*(cell_self_weight.y)*(cell_self_weight.z);
	atomicAdd(&volume(cell_idx.x + 1, cell_idx.y, cell_idx.z), temp);
	//0,1,0
	temp = (cell_self_weight.x)*(cell_neigh_weight.y)*(cell_self_weight.z);
	atomicAdd(&volume(cell_idx.x, cell_idx.y + 1, cell_idx.z), temp);
	//1,1,0
	temp = (cell_neigh_weight.x)*(cell_neigh_weight.y)*(cell_self_weight.z);
	atomicAdd(&volume(cell_idx.x + 1, cell_idx.y + 1, cell_idx.z), temp);
	//0,0,1
	temp = (cell_self_weight.x)*(cell_self_weight.y)*(cell_neigh_weight.z);
	atomicAdd(&volume(cell_idx.x, cell_idx.y, cell_idx.z + 1), temp);
	//1,0,1
	temp = (cell_neigh_weight.x)*(cell_self_weight.y)*(cell_neigh_weight.z);
	atomicAdd(&volume(cell_idx.x + 1, cell_idx.y, cell_idx.z + 1), temp);
	//0,1,1
	temp = (cell_self_weight.x)*(cell_neigh_weight.y)*(cell_neigh_weight.z);
	atomicAdd(&volume(cell_idx.x, cell_idx.y + 1, cell_idx.z + 1), temp);
	//1,1,1
	temp = (cell_neigh_weight.x)*(cell_neigh_weight.y)*(cell_neigh_weight.z);
	atomicAdd(&volume(cell_idx.x + 1, cell_idx.y + 1, cell_idx.z + 1), temp);

}

template <typename CONST_SAMPLES_IT_TYPE>
void VolumeBinningEstimator::estimateBinnedVolumeFromIterator(Volume& volume, CONST_SAMPLES_IT_TYPE samples_device, int nsamples, bool linear/*=false*/)
{
	int launch_block_size = 512;
	int launch_grid_size = (nsamples + launch_block_size - 1) / launch_block_size;

	GPUVolumeData volData = volume.getVolumeData();
	if (linear)
		buildLinearBins3d <<<launch_grid_size, launch_block_size >>> (samples_device, volData, nsamples);
	else
		buildBins3d <<<launch_grid_size, launch_block_size >>> (samples_device, volData, nsamples);
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());
}