#include "KernelDensityEstimationCUDA.h"
#include "Volume.h"
#include "KernelBandwidth.h"

#include <helper_math.h>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include <memory>
#include <algorithm>

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

#include "GaussianKernel.cuh"

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
						val += GaussianKernel(pos , samples[hashSampleIdxVec[neigh_sample_idx].sampleIdx]);
						++neigh_sample_idx;
					}
				}
			}

	volume(idx_x, idx_y, idx_z) += val;
}

void KernelDensityEstimationCUDA::estimateAccurate(Volume& volume, const float4* samples_device, int nsamples, const KernelBandwidth& kernelBandwidth)
{
	float H = sqrtf(kernelBandwidth.getLargestEigenValue());
	float4 hashCellSize_ = make_float4((3.7f*H), (3.7f*H), (3.7f*H), 1.0);
	float4 minBB, maxBB;
	
	volume.getBox(minBB, maxBB);

	cudaMemcpyToSymbol(hashCellSize, &hashCellSize_, sizeof(float4));
	cudaMemcpyToSymbol(hashOffset, &minBB, sizeof(float4));
	

	uint4 hashGridSize_ = make_uint4(::ceilf((maxBB.x - minBB.x) / hashCellSize_.x), ::ceilf((maxBB.y - minBB.y) / hashCellSize_.y), ::ceilf((maxBB.z - minBB.z) / hashCellSize_.z), 0);
	cudaMemcpyToSymbol(hashGridSize, &hashGridSize_, sizeof(uint4));

	thrust::device_vector<HashSampleIdxStruct> hashSampleIdxVec_device(nsamples);

	int launch_block_size = 256;
	int launch_grid_size = (nsamples + launch_block_size - 1) / launch_block_size;

	assignHashForSamples <<<launch_grid_size, launch_block_size >>> (samples_device, thrust::raw_pointer_cast(&hashSampleIdxVec_device[0]), nsamples);

	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());

	thrust::sort(hashSampleIdxVec_device.begin(), hashSampleIdxVec_device.end(), hashSampleIdxComparer());

	int hashTableSize = hashGridSize_.x*hashGridSize_.y*hashGridSize_.z;

	thrust::device_vector<int> hashStartIdxLT(hashTableSize, -1);

	fillHashLookupTable<<<launch_grid_size, launch_block_size >>>(thrust::raw_pointer_cast(&hashSampleIdxVec_device[0]), thrust::raw_pointer_cast(&hashStartIdxLT[0]), nsamples);
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());

	SymmetricMatrix H_inv_ = kernelBandwidth.getInverseMatrix();

	chkCudaErrors(cudaMemcpyToSymbol(H_inv, &H_inv_, sizeof(SymmetricMatrix)));

	float kernel_normalizing_constant_ = 0.0634936359342409 //1/(2 pi)^(3/2)
		/ (sqrtf(kernelBandwidth.getDeterminant()));
	chkCudaErrors(cudaMemcpyToSymbol(kernel_normalizing_constant, &kernel_normalizing_constant_, sizeof(float)));

	launch_block_size = 512;
	launch_grid_size = (volume.numberOfCells() + launch_block_size - 1) / launch_block_size;
	
	GPUVolumeData volData = volume.getVolumeData();

	estimateKDEAccurateWithSpatialHash_kernel <<<launch_grid_size, launch_block_size >>> (thrust::raw_pointer_cast(&samples_device[0]), nsamples,
		thrust::raw_pointer_cast(&hashSampleIdxVec_device[0]), thrust::raw_pointer_cast(&hashStartIdxLT[0]),
		volData
		);
   
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());
}



template <class Kernel>
__global__ void fillKernelFilter(GPUVolumeData kernelFilterData,Kernel kernel)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint3 gridSize = kernelFilterData.gridSize();
	if (idx >= gridSize.x*gridSize.y*gridSize.z) return;

	int idx_x = idx % gridSize.x;
	int idx_y = (idx / gridSize.x) % gridSize.y;
	int idx_z = idx / (gridSize.y * gridSize.x);

	float3 pos = make_float3(kernelFilterData.cellCenter(idx_x, idx_y, idx_z));

	kernelFilterData(idx_x, idx_y, idx_z) = kernel(pos);

}

__constant__ float fftScaleFactor;

__global__ void filterInFreqDomain(cuFloatComplex* __restrict__ Fdata, const cuFloatComplex* __restrict__ Ffilter , const int data_size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= data_size) return;
	Fdata[idx] = cuCmulf(Fdata[idx], Ffilter[idx]);
	Fdata[idx] *= fftScaleFactor;
}


#include "VolumeBinning.h"
#include "VolumeBinning.cuh"

void VolumeBinningEstimator::estimateBinnedVolume(Volume& volume, const float4* __restrict__ samples_device, int nsamples, bool linear/*=false*/)
{
	estimateBinnedVolumeFromIterator<const float4*>(volume, samples_device, nsamples, linear);
}

void KernelDensityEstimationCUDA::estimateBinned(Volume& volume, const float4* samples_device, int nsamples, const KernelBandwidth& kernelBandwidth)
{

	VolumeBinningEstimator::estimateBinnedVolume(volume, samples_device, nsamples);
	estimateBinned(volume, kernelBandwidth);
}

#include <cufft.h>
#include "FFTKDECUDAMemoryManager.h"
#include "FFTKDEEstimationBaseCUDA.h"

template <class KernelFiller>
void FFTKDEEstimationBaseCUDA::estimateBinnedBase(GPUVolumeData& volData, int3& L, int3& P, const KernelBandwidth& kernelBandwidth)
{
	unsigned int grid_element_num = P.x*P.y*P.z;
	//thrust::device_vector<float> Czp(P.x*P.y*P.z);
	memoryManager->allocate(P);

	chkCudaErrors(cudaMemset(memoryManager->Czp_data(), 0, grid_element_num * sizeof(float)));
	{
		cudaMemcpy3DParms myParms = { 0 };
		myParms.srcPtr = volData.pitchedDevPtr;
		myParms.dstPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(memoryManager->Czp_data()), P.x * sizeof(float), P.x * sizeof(float), P.y);
		myParms.extent = make_cudaExtent(volData.width_ * sizeof(float), volData.height_, volData.depth_);
		myParms.dstPos = make_cudaPos((L.x) * sizeof(float), L.y, L.z);
		myParms.kind = cudaMemcpyDeviceToDevice;

		chkCudaErrors(cudaMemcpy3D(&myParms));
	}

	//thrust::device_vector<float> Kzp(P.x*P.y*P.z);
	chkCudaErrors(cudaMemset(memoryManager->Kzp_data(), 0, grid_element_num * sizeof(float)));

	GPUVolumeData kernelFilterData;
	kernelFilterData.pitchedDevPtr = make_cudaPitchedPtr(memoryManager->Kzp_data(), P.x * sizeof(float), P.x * sizeof(float), P.y);
	kernelFilterData.width_ = (2 * L.x + 1);
	kernelFilterData.height_ = (2 * L.y + 1);
	kernelFilterData.depth_ = (2 * L.z + 1);
	kernelFilterData.cellSize_ = volData.cellSize_;
	kernelFilterData.BB_.lower = make_float4(-(L.x + 0.5f)*kernelFilterData.cellSize_.x, -(L.y + 0.5f)*kernelFilterData.cellSize_.y, -(L.z + 0.5f)*kernelFilterData.cellSize_.z, 0.0f);
	kernelFilterData.BB_.upper = make_float4((L.x + 0.5f)*kernelFilterData.cellSize_.x, (L.y + 0.5f)*kernelFilterData.cellSize_.y, (L.z + 0.5f)*kernelFilterData.cellSize_.z, 0.0f);

	SymmetricMatrix H_inv_ = kernelBandwidth.getInverseMatrix();

	chkCudaErrors(cudaMemcpyToSymbol(H_inv, &H_inv_, sizeof(SymmetricMatrix)));

	kernel_normalizing_constant_ = 0.0634936359342409 //1/(2 pi)^(3/2)
		/ (sqrtf(kernelBandwidth.getDeterminant()));
	chkCudaErrors(cudaMemcpyToSymbol(kernel_normalizing_constant, &kernel_normalizing_constant_, sizeof(float)));

	int launch_block_size = 512;
	int launch_grid_size = (kernelFilterData.width_*kernelFilterData.height_*kernelFilterData.depth_ + launch_block_size - 1) / launch_block_size;

	fillKernelFilter<KernelFiller> <<<launch_grid_size, launch_block_size >>> (kernelFilterData, KernelFiller());
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());



	cufftHandle forwardPlan, inversePlan;
	chkCudaErrors(cufftCreate(&forwardPlan));
	chkCudaErrors(cufftCreate(&inversePlan));
	chkCudaErrors(cufftSetAutoAllocation(forwardPlan,false));
	chkCudaErrors(cufftSetAutoAllocation(inversePlan, false));

	size_t fftWorkAreaForward = 0;
	size_t fftWorkAreaInverse = 0;
	
	chkCudaErrors(cufftMakePlan3d(forwardPlan, P.z, P.y, P.x, cufftType::CUFFT_R2C, &fftWorkAreaForward));	
	chkCudaErrors(cufftMakePlan3d(inversePlan, P.z, P.y, P.x, cufftType::CUFFT_C2R, &fftWorkAreaInverse));

	memoryManager->allocateFFTWorkArea(std::max<std::size_t>( fftWorkAreaForward,fftWorkAreaInverse) );
	chkCudaErrors(cufftSetWorkArea(forwardPlan, memoryManager->fftWorkArea()));
	chkCudaErrors(cufftSetWorkArea(inversePlan, memoryManager->fftWorkArea()));

	//thrust::device_vector<cuFloatComplex> FCzp((P.x / 2 + 1)*P.y*P.z);
	//thrust::device_vector<cuFloatComplex> FKzp((P.x / 2 + 1)*P.y*P.z);

	chkCudaErrors(cufftExecR2C(forwardPlan, memoryManager->Czp_data(), memoryManager->FCzp_data()));
	chkCudaErrors(cufftExecR2C(forwardPlan, memoryManager->Kzp_data(), memoryManager->FKzp_data()));

	launch_block_size = 512;
	launch_grid_size = (grid_element_num + launch_block_size - 1) / launch_block_size;

	float fftScaleFactor_ = 1.0f / ((grid_element_num));
	chkCudaErrors(cudaMemcpyToSymbol(fftScaleFactor, &fftScaleFactor_, sizeof(float)));

	filterInFreqDomain <<< launch_grid_size, launch_block_size >>> (memoryManager->FCzp_data(), memoryManager->FKzp_data(), grid_element_num);

	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());	
	
	chkCudaErrors(cufftDestroy(forwardPlan));

	//chkCudaErrors(cufftPlan3d(&inversePlan, P.z, P.y, P.x, cufftType::CUFFT_C2R));
	chkCudaErrors(cufftExecC2R(inversePlan, memoryManager->FCzp_data(), memoryManager->Czp_data()));
	

	chkCudaErrors(cufftDestroy(inversePlan));
}



struct GaussianKernelFiller
{
	__device__ float operator()(float3 pos)
	{
		return GaussianKernel(pos, make_float3(0.0f));
	}
};


void KernelDensityEstimationCUDA::estimateBinned(Volume& volume, const KernelBandwidth& kernelBandwidth)
{
	GPUVolumeData volData = volume.getVolumeData();

	float3 L_f = make_float3((3.7f*sqrtf(kernelBandwidth.getLargestEigenValue()))) / volume.cellSize();
	L_f = make_float3(ceilf(L_f.x), ceilf(L_f.y), ceilf(L_f.z));
	int3 L = make_int3(L_f.x, L_f.y, L_f.z);

	int3 P = make_int3(powf(2.0f, ceilf(log2f(L_f.x*2.0f + 1.0f + volData.width_))), powf(2.0f, ceilf(log2f(L_f.y*2.0f + 1.0f + volData.height_))), powf(2.0f, ceilf(log2f(L_f.z*2.0f + 1.0f + volData.depth_))));

	estimateBinnedBase<GaussianKernelFiller>(volData,L,P, kernelBandwidth);

	{
		cudaMemcpy3DParms myParms = { 0 };
		myParms.dstPtr = volume.getVolumeData().pitchedDevPtr;
		myParms.srcPtr = make_cudaPitchedPtr(memoryManager->Czp_data(), P.x * sizeof(float), P.x * sizeof(float), P.y);
		myParms.extent = make_cudaExtent(volData.width_ * sizeof(float), volData.height_, volData.depth_);
		myParms.srcPos = make_cudaPos((L.x*2)*sizeof(float), L.y*2, L.z*2);
		myParms.kind = cudaMemcpyDeviceToDevice;

		chkCudaErrors(cudaMemcpy3D(&myParms));
	}
//std::vector<float> debug(P.x*P.y*P.z);
//std::vector<float> debug2(P.x*P.y*P.z);
//	thrust::copy_n(Czp.begin(), P.x*P.y*P.z, debug.begin());
//	thrust::copy_n(Kzp.begin(), P.x*P.y*P.z, debug2.begin());
//	printf("x,y,z,density\n");
//	int idx = 0;
//	for (int k = 0; k<P.z; ++k)
//		for (int j = 0; j<P.y; ++j)
//			for (int i = 0; i < P.x; ++i)
//			{
//				float3 cellPos = (make_float3(float(i) + 0.5f, float(j) + 0.5f, float(k) + 0.5f)-make_float3((L.x)*2.0f,(L.y)*2.0f,(L.z)*2.0f))*make_float3(volData.cellSize_) + make_float3(volData.minBB_);
//				printf("%g,%g,%g,%g\n", cellPos.x, cellPos.y, cellPos.z, debug[idx]);
//				++idx;
//			}
//	

}

#include "FFTLSCVEstimatorCUDA.h"


struct BoundingBox_reduction : public thrust::binary_function<BoundingBox, BoundingBox, BoundingBox>
{
	__host__ __device__
		BoundingBox operator()(BoundingBox a, BoundingBox b)
	{
		// lower left corner
		float4 ll=make_float4(thrust::min(a.lower.x, b.lower.x), thrust::min(a.lower.y, b.lower.y), thrust::min(a.lower.z, b.lower.z),0.0f);

		// upper right corner
		float4 ur=make_float4(thrust::max(a.upper.x, b.upper.x), thrust::max(a.upper.y, b.upper.y), thrust::max(a.upper.z, b.upper.z),0.0f);

		return BoundingBox(ll, ur);
	}
};

BoundingBox estimateBoundingBox(const float4* samples_device, int nsamples)
{
	float4 frst;
	chkCudaErrors(cudaMemcpy(&frst, samples_device, sizeof(float4), cudaMemcpyDeviceToHost));
	BoundingBox init(frst);
	BoundingBox_reduction binary_op;
	BoundingBox result = thrust::reduce(thrust::device_ptr<const float4>(samples_device), thrust::device_ptr<const float4>(samples_device+nsamples), init, binary_op);
	return result;
}


struct LSCVKernelFiller
{
	__device__ float operator()(float3 pos)
	{
		return GaussianKernel(make_float4(pos, 0.0), make_float4(0.0f, 0.0f, 0.0f, 2.0f)) - 2.0f*GaussianKernel(pos, make_float3(0.0f, 0.0f, 0.0f));
	}
};

#include "PitchedMemoryIterator.cuh"

FFTLSCVEstimatorCUDA::FFTLSCVEstimatorCUDA():
	FFTKDEEstimationBaseCUDA()
{
}

float FFTLSCVEstimatorCUDA::estimateLSCV(const Volume& binned_volume, int nsamples, const KernelBandwidth& kernelBandwidth)
{
	GPUVolumeData volData = binned_volume.getVolumeData();
	float3 L_f = make_float3((3.7f*sqrtf(kernelBandwidth.getLargestEigenValue()))) / binned_volume.cellSize()* 2.0;
	L_f = make_float3(ceilf(L_f.x), ceilf(L_f.y), ceilf(L_f.z));
	int3 L = make_int3(L_f.x, L_f.y, L_f.z);

	int3 P = make_int3(powf(2.0f, ceilf(log2f(L_f.x*2.0f + 1.0f + volData.width_))), powf(2.0f, ceilf(log2f(L_f.y*2.0f + 1.0f + volData.height_))), powf(2.0f, ceilf(log2f(L_f.z*2.0f + 1.0f + volData.depth_))));

	estimateBinnedBase<LSCVKernelFiller>(volData, L, P, kernelBandwidth);

	thrust_ext::PitchedMemoryIterator<float> pIt(make_cudaPitchedPtr(memoryManager->Czp_data(), sizeof(float)*P.x, sizeof(float)*P.x, P.y), make_cudaExtent(volData.width_, volData.height_, volData.depth_), make_cudaPos((L.x * 2), L.y * 2, L.z * 2));
	thrust_ext::PitchedMemoryIterator<float> volDataIt(volData.pitchedDevPtr);
	
	thrust::transform(pIt, pIt + volData.width_*volData.height_*volData.depth_, volDataIt, pIt, thrust::placeholders::_1*thrust::placeholders::_2);
	register float temp = thrust::reduce(pIt, pIt + volData.width_*volData.height_*volData.depth_, 0.0f);
	return (temp / float(nsamples) + 2.0f*kernel_normalizing_constant_) / float(nsamples);
}
/*
float FFTLSCVEstimatorCUDA::estimateLSCV(const Volume& binned_volume,int nsamples,const KernelBandwidth& kernelBandwidth)
{
	GPUVolumeData volData = binned_volume.getVolumeData();
	float3 L_f = make_float3((3.7f*sqrtf(kernelBandwidth.getLargestEigenValue()))) / binned_volume.cellSize()* 2.0;
	L_f = make_float3(ceilf(L_f.x), ceilf(L_f.y), ceilf(L_f.z));
	int3 L = make_int3(L_f.x, L_f.y, L_f.z);
	
	int3 P = make_int3(powf(2.0f, ceilf(log2f(L_f.x*2.0f + 1.0f + volData.width_))), powf(2.0f, ceilf(log2f(L_f.y*2.0f + 1.0f + volData.height_))), powf(2.0f, ceilf(log2f(L_f.z*2.0f + 1.0f + volData.depth_))));

	thrust::device_vector<float> Czp(P.x*P.y*P.z);
	chkCudaErrors(cudaMemset(thrust::raw_pointer_cast(Czp.data()), 0, P.x*P.y*P.z * sizeof(float)));
	{
		cudaMemcpy3DParms myParms = { 0 };
		myParms.srcPtr = binned_volume.getVolumeData().pitchedDevPtr;
		myParms.dstPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(Czp.data()), P.x * sizeof(float), P.x * sizeof(float), P.y);
		myParms.extent = make_cudaExtent(volData.width_ * sizeof(float), volData.height_, volData.depth_);
		myParms.dstPos = make_cudaPos(L.x * sizeof(float), L.y, L.z);
		myParms.kind = cudaMemcpyDeviceToDevice;

		chkCudaErrors(cudaMemcpy3D(&myParms));
	}

	thrust::device_vector<float> Kzp(P.x*P.y*P.z);
	chkCudaErrors(cudaMemset(thrust::raw_pointer_cast(Kzp.data()), 0, P.x*P.y*P.z * sizeof(float)));

	GPUVolumeData kernelFilterData;
	kernelFilterData.pitchedDevPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(Kzp.data()), P.x * sizeof(float), P.x * sizeof(float), P.y);
	kernelFilterData.width_ = (2 * L.x + 1);
	kernelFilterData.height_ = (2 * L.y + 1);
	kernelFilterData.depth_ = (2 * L.z + 1);
	kernelFilterData.cellSize_ = volData.cellSize_;
	kernelFilterData.BB_.lower = make_float4(-(L.x + 0.5f)*kernelFilterData.cellSize_.x, -(L.y + 0.5f)*kernelFilterData.cellSize_.y, -(L.z + 0.5f)*kernelFilterData.cellSize_.z, 0.0f);
	kernelFilterData.BB_.upper = make_float4((L.x + 0.5f)*kernelFilterData.cellSize_.x, (L.y + 0.5f)*kernelFilterData.cellSize_.y, (L.z + 0.5f)*kernelFilterData.cellSize_.z, 0.0f);

	SymmetricMatrix H_inv_ = kernelBandwidth.getInverseMatrix();

	chkCudaErrors(cudaMemcpyToSymbol(H_inv, &H_inv_, sizeof(SymmetricMatrix)));

	float kernel_normalizing_constant_ = 0.0634936359342409 //1/(2 pi)^(3/2)
		/ (sqrtf(kernelBandwidth.getDeterminant()));
	chkCudaErrors(cudaMemcpyToSymbol(kernel_normalizing_constant, &kernel_normalizing_constant_, sizeof(float)));

	int launch_block_size = 512;
	int launch_grid_size = (kernelFilterData.width_*kernelFilterData.height_*kernelFilterData.depth_ + launch_block_size - 1) / launch_block_size;

	fillLSCVKernelFilter <<<launch_grid_size, launch_block_size >>> (kernelFilterData);
	//fillKernelFilter <<<launch_grid_size, launch_block_size >>> (kernelFilterData);
	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());



	cufftHandle forwardPlan, inversePlan;
	chkCudaErrors(cufftPlan3d(&forwardPlan, P.z, P.y, P.x, cufftType::CUFFT_R2C));
	chkCudaErrors(cufftPlan3d(&inversePlan, P.z, P.y, P.x, cufftType::CUFFT_C2R));

	thrust::device_vector<cuFloatComplex> FCzp((P.x / 2 + 1)*P.y*P.z);
	thrust::device_vector<cuFloatComplex> FKzp((P.x / 2 + 1)*P.y*P.z);

	chkCudaErrors(cufftExecR2C(forwardPlan, thrust::raw_pointer_cast(Czp.data()), thrust::raw_pointer_cast(FCzp.data())));
	chkCudaErrors(cufftExecR2C(forwardPlan, thrust::raw_pointer_cast(Kzp.data()), thrust::raw_pointer_cast(FKzp.data())));

	launch_block_size = 512;
	launch_grid_size = (FCzp.size() + launch_block_size - 1) / launch_block_size;

	float fftScaleFactor_ = 1.0f / ((P.x*P.y*P.z));
	chkCudaErrors(cudaMemcpyToSymbol(fftScaleFactor, &fftScaleFactor_, sizeof(float)));

	filterInFreqDomain <<< launch_grid_size, launch_block_size >>> (thrust::raw_pointer_cast(FCzp.data()), thrust::raw_pointer_cast(FKzp.data()), FCzp.size());

	cudaDeviceSynchronize();
	chkCudaErrors(cudaGetLastError());

	chkCudaErrors(cufftExecC2R(inversePlan, thrust::raw_pointer_cast(FCzp.data()), thrust::raw_pointer_cast(Czp.data())));

    //thrust::device_vector<float> dev_debug(volData.width_ *  volData.height_* volData.depth_);
	//{
	//	cudaMemcpy3DParms myParms = { 0 };
	//	myParms.dstPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(dev_debug.data()), volData.width_ * sizeof(float), volData.width_ * sizeof(float), volData.height_);
	//	myParms.srcPtr = make_cudaPitchedPtr(thrust::raw_pointer_cast(Czp.data()), P.x * sizeof(float), P.x * sizeof(float), P.y);
	//	myParms.extent = make_cudaExtent(volData.width_ * sizeof(float), volData.height_, volData.depth_);
	//	myParms.srcPos = make_cudaPos((L.x * 2) * sizeof(float), L.y * 2, L.z * 2);
	//	myParms.kind = cudaMemcpyDeviceToDevice;
	//
	//	chkCudaErrors(cudaMemcpy3D(&myParms));
	//}
	
	thrust_ext::PitchedMemoryIterator<float> pIt(make_cudaPitchedPtr(thrust::raw_pointer_cast(Czp.data()),sizeof(float)*P.x, sizeof(float)*P.x,P.y), make_cudaExtent(volData.width_, volData.height_, volData.depth_), make_cudaPos((L.x * 2), L.y * 2, L.z * 2));
	//std::vector<float> debug3(dev_debug.size());
	thrust_ext::PitchedMemoryIterator<float> volDataIt(volData.pitchedDevPtr);

	//thrust::copy_n(pIt, dev_debug.size(), dev_debug.begin());
	//thrust::copy_n(dev_debug.begin(), dev_debug.size(), debug3.begin());
	//std::vector<float> debug2(P.x*P.y*P.z);
	//thrust::copy_n(volDataIt, dev_debug.size(), dev_debug.begin());
	//thrust::copy_n(dev_debug.begin(), dev_debug.size(), debug2.begin());

	//std::vector<float> debug2(P.x*P.y*P.z);
	//	thrust::copy_n(Czp.begin(), P.x*P.y*P.z, debug.begin());
	//	thrust::copy_n(Kzp.begin(), P.x*P.y*P.z, debug2.begin());
	//	printf("x,y,z,density\n");
	//	int idx = 0;
	//	float tmp_ = 0.0f;
	//	for (int k = 0; k<volData.depth_; ++k)
	//		for (int j = 0; j<volData.height_; ++j)
	//			for (int i = 0; i < volData.width_; ++i)
	//			{
	//				float3 cellPos = (make_float3(float(i) + 0.5f, float(j) + 0.5f, float(k) + 0.5f))*make_float3(volData.cellSize_) + make_float3(volData.minBB_);
	//				printf("%g,%g,%g,%g\n", cellPos.x, cellPos.y, cellPos.z, debug3[idx]* debug2[idx]);
	//				
	//				tmp_ += debug3[idx] *debug2[idx];
	//				
	//				++idx;
	//			}

	//std::vector<float> debug(P.x*P.y*P.z);
	//thrust::copy_n(Czp.begin(), P.x*P.y*P.z, debug.begin());
	//	printf("x,y,z,density\n");
	//	int idx = 0;
	//	for (int k = 0; k<P.z; ++k)
	//		for (int j = 0; j<P.y; ++j)
	//			for (int i = 0; i < P.x; ++i)
	//			{
	//				float3 cellPos = (make_float3(float(i) + 0.5f, float(j) + 0.5f, float(k) + 0.5f)-make_float3((L.x)*2.0f-2.0,(L.y)*2.0f-2.0,(L.z)*2.0f-2.0))*make_float3(volData.cellSize_) + make_float3(volData.BB_.lower);
	//				printf("%g,%g,%g,%g\n", cellPos.x, cellPos.y, cellPos.z, debug[idx]);
	//				++idx;
	//			}
	
		thrust::transform(pIt, pIt + volData.width_*volData.height_*volData.depth_, volDataIt, pIt, thrust::placeholders::_1*thrust::placeholders::_2);
		register float temp = thrust::reduce(pIt, pIt + volData.width_*volData.height_*volData.depth_, 0.0f);
		

	chkCudaErrors(cufftDestroy(forwardPlan));
	chkCudaErrors(cufftDestroy(inversePlan));

	return (temp / float(nsamples) + 2.0f*kernel_normalizing_constant_) / float(nsamples);
}

*/