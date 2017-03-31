#include "KernelDensityEstimationCUDA.h"
#include "Volume.h"
#include "KernelBandwidth.h"

#include <helper_math.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <memory>
//
//class KernelDensityEstimationCUDA
//{
//public:
//	KernelDensityEstimationCUDA();
//	void setSamplesFromHost(const float4* samples_host, int N);
//	
//	void setVolume(std::shared_ptr<Volume>);
//	std::weak_ptr<Volume> getVolume() const;
//
//	void setKernelBandwidth(std::shared_ptr<KernelBandwidth>);
//	void estimateAccurate();
//
//private:
//	//float4* samples_dev;
//	thrust::device_vector<float4> samples_device;
//	int nsamples;
//	
//	std::shared_ptr<Volume> volume;
//	std::shared_ptr<KernelBandwidth> kernelBandwidth;
//};

KernelDensityEstimationCUDA::KernelDensityEstimationCUDA():
	nsamples(0)
	//samples_dev(nullptr)
{

}

void KernelDensityEstimationCUDA::setSamplesFromHost(const float4* samples_host, int N)
{
	samples_device.resize(N);
	thrust::copy_n(samples_host, N, samples_device.begin());
	nsamples = N;
}

void  KernelDensityEstimationCUDA::setVolume(std::shared_ptr<Volume> v)
{
	volume=v;
}

std::weak_ptr<Volume>  KernelDensityEstimationCUDA::getVolume() const
{
	return std::weak_ptr<Volume>(volume);
}

void KernelDensityEstimationCUDA::setKernelBandwidth(std::shared_ptr<KernelBandwidth> kb)
{
	kernelBandwidth = kb;
}

__constant__ float4 hashCellSize;
__constant__ uint4 hashGridSize;
__constant__ float4 hashOffset;

__device__ int4 hashGridIdx(float4 position)
{
	register float4 tmp = floorf((position - hashOffset) / hashCellSize);
	return make_int4(tmp.x,tmp.y,tmp.z,tmp.w);
}

//__constant__ int numBuckets = 1024;
//
//
//// Expands a 10 - bit integer into 30 bits
//// by inserting 2 zeros after each bit.
//__device__ unsigned int expandBits(unsigned int v)
//{
//	v = (v * 0x00010001u) & 0xFF0000FFu;
//	v = (v * 0x00000101u) & 0x0F00F00Fu;
//	v = (v * 0x00000011u) & 0xC30C30C3u;
//	v = (v * 0x00000005u) & 0x49249249u;
//	return v;
//}
//
//__device__ unsigned int morton3D(int4 grid_idx)
//{
//	unsigned int xx = expandBits((unsigned int)grid_idx.x);
//	unsigned int yy = expandBits((unsigned int)grid_idx.y);
//	unsigned int zz = expandBits((unsigned int)grid_idx.z);
//	return (xx * 4 + yy * 2 + zz);
//}

__device__ unsigned int spatialHash(int4 hash_grid_idx)
{	
	return (hash_grid_idx.z*hashGridSize.y + hash_grid_idx.y)*hashGridSize.x + hash_grid_idx.x;
	//const uint p1 = 73856093; // some large primes
	//const uint p2 = 19349663;
	//const uint p3 = 83492791;
	//int n = p1*hash_grid_idx.x ^ p2*hash_grid_idx.y ^ p3*hash_grid_idx.z;
	//n %= numBuckets;
	//return n;
	//return morton3D(hash_grid_idx)%numBuckets;
}
struct HashSampleIdxStruct
{
	unsigned int hash;
	unsigned int sampleIdx;
};

struct hashSampleIdxComparer
{
	__device__ bool operator()(const HashSampleIdxStruct& l, const HashSampleIdxStruct& r) const
	{
		return l.hash < r.hash;
	}
};

__global__ void assignHashForSamples(const float4* __restrict__ samples, HashSampleIdxStruct* __restrict__ hashSampleIdxVec, const int nsamples)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= nsamples) return;
	hashSampleIdxVec[idx].hash = spatialHash(hashGridIdx(samples[idx]));
	hashSampleIdxVec[idx].sampleIdx = idx;
}

__global__ void fillHashLookupTable(const HashSampleIdxStruct* __restrict__ hashSampleIdx, int* __restrict__ hashStartIdx, const int nsamples)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= nsamples) return;
	if (idx == 0 || hashSampleIdx[idx - 1].hash != hashSampleIdx[idx].hash) hashStartIdx[hashSampleIdx[idx].hash] = idx;
}

__constant__ SymmetricMatrix H_inv;

__constant__ float kernel_normalizing_constant;

__device__ float GaussianKernel(float4 pos)
{
	register float3 tmp = H_inv.mult(make_float3(pos));
	
	return kernel_normalizing_constant*exp(-0.5f*dot(make_float3(pos),tmp));
}

__global__ void estimateKDEAccurateWithSpatialHash_kernel(const float4* __restrict__ samples, const int nsamples, const HashSampleIdxStruct* __restrict__ hashSampleIdxVec,const int* __restrict__ hashStartIdx/*,Volume volume*/
	, GPUVolumeData volume
)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= volume.numberOfCells()) return;
	uint3 gridSize = volume.gridSize();

	int idx_x = idx % gridSize.y;
	int idx_y = (idx / gridSize.y) % gridSize.z;
	int idx_z = idx / (gridSize.y * gridSize.z);
	
	float4 pos = volume.cellCenter(idx_x, idx_y, idx_z);
	int4 hash_grid_idx = hashGridIdx(pos);
	

	float val = 0.0;
	unsigned int hash;

#pragma unroll
	for (int du=-1;du<=1;++du)
#pragma unroll
		for (int dv = -1 ; dv <= 1; ++dv)
#pragma unroll
			for (int dk = -1 ; dk <= 1; ++dk)
			{
				int4 hash_neigh_grid_idx = hash_grid_idx + make_int4(du, dv, dk, 0);
				if ((hash_neigh_grid_idx.x >= 0) && (hash_neigh_grid_idx.x < hashGridSize.x) &&
					(hash_neigh_grid_idx.y >= 0) && (hash_neigh_grid_idx.y < hashGridSize.y) &&
					(hash_neigh_grid_idx.z >= 0) && (hash_neigh_grid_idx.z < hashGridSize.z))
				{
					hash = spatialHash(hash_neigh_grid_idx);
					
					int neigh_sample_idx = hashStartIdx[hash];
					if (neigh_sample_idx == -1) continue;
					
					while (neigh_sample_idx < nsamples && hashSampleIdxVec[neigh_sample_idx].hash == hash)
					{
						val += GaussianKernel(pos - samples[hashSampleIdxVec[neigh_sample_idx].sampleIdx]);
						++neigh_sample_idx;
					}
				}
			}

	volume(idx_x, idx_y, idx_z) += val/float(nsamples);
	
	
}

void KernelDensityEstimationCUDA::estimateAccurate()
{
	float H = sqrtf(kernelBandwidth->getLargestEigenValue());
	float4 hashCellSize_ = make_float4((4.0f*H), (4.0f*H), (4.0f*H), 1.0);
	float4 minBB, maxBB;
	volume->getBox(minBB, maxBB);

	cudaMemcpyToSymbol(hashCellSize, &hashCellSize_, sizeof(float4));
	cudaMemcpyToSymbol(hashOffset, &minBB, sizeof(float4));
	

	uint4 hashGridSize_ = make_uint4(::ceilf((maxBB.x - minBB.x) / hashCellSize_.x), ::ceilf((maxBB.y - minBB.y) / hashCellSize_.y), ::ceilf((maxBB.z - minBB.z) / hashCellSize_.z), 0);
	cudaMemcpyToSymbol(hashGridSize, &hashGridSize_, sizeof(uint4));

	thrust::device_vector<HashSampleIdxStruct> hashSampleIdxVec_device(nsamples);

	int launch_block_size = 256;
	int launch_grid_size = (nsamples + launch_block_size - 1) / launch_block_size;

	assignHashForSamples <<<launch_grid_size, launch_block_size >>> (thrust::raw_pointer_cast(&samples_device[0]), thrust::raw_pointer_cast(&hashSampleIdxVec_device[0]), nsamples);

	cudaDeviceSynchronize();
	chkCudaErrors(/*cudaGetLastError()*/cudaErrorUnknown);

	thrust::sort(hashSampleIdxVec_device.begin(), hashSampleIdxVec_device.end(), hashSampleIdxComparer());

	int hashTableSize = hashGridSize_.x*hashGridSize_.y*hashGridSize_.z;

	thrust::device_vector<int> hashStartIdxLT(hashTableSize, -1);

	fillHashLookupTable<<<launch_grid_size, launch_block_size >>>(thrust::raw_pointer_cast(&hashSampleIdxVec_device[0]), thrust::raw_pointer_cast(&hashStartIdxLT[0]), nsamples);
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());

	std::vector<HashSampleIdxStruct> debug(nsamples);
	thrust::copy_n(hashSampleIdxVec_device.begin(), nsamples, debug.begin());

	std::vector<int> debug2(hashTableSize);
	thrust::copy_n(hashStartIdxLT.begin(), hashTableSize, debug2.begin());


	SymmetricMatrix H_inv_ = kernelBandwidth->getInverseMatrix();

	chkCudaErrors(cudaMemcpyToSymbol(H_inv, &H_inv_, sizeof(SymmetricMatrix)));

	float kernel_normalizing_constant_ = 0.0634936359342409 //1/(2 pi)^(3/2)
		/ (sqrtf(kernelBandwidth->getDeterminant()));
	chkCudaErrors(cudaMemcpyToSymbol(kernel_normalizing_constant, &kernel_normalizing_constant_, sizeof(float)));

	launch_block_size = 256;
	launch_grid_size = (volume->numberOfCells() + launch_block_size - 1) / launch_block_size;
	
	GPUVolumeData volData = volume->getVolumeData();

	estimateKDEAccurateWithSpatialHash_kernel <<<launch_grid_size, launch_block_size >>> (thrust::raw_pointer_cast(&samples_device[0]), nsamples,
		thrust::raw_pointer_cast(&hashSampleIdxVec_device[0]), thrust::raw_pointer_cast(&hashStartIdxLT[0]),
		volData
		);
   
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());
}

