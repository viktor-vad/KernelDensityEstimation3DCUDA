#pragma once

#include <cuda_runtime.h>
//#include <thrust/device_vector.h>


#include <thrust/iterator/iterator_facade.h>


namespace thrust_ext
{


	
	template <typename T>
	struct PitchedMemoryIterator :public  thrust::iterator_facade<PitchedMemoryIterator<T>, T, thrust::device_system_tag, thrust::random_access_traversal_tag, T&>
	{
	public:
		explicit PitchedMemoryIterator(cudaPitchedPtr pitchedPtr)
			:pitchedPtr_(pitchedPtr),
			extent_(make_cudaExtent(pitchedPtr.xsize,pitchedPtr.ysize,1))
		{

		}
		
		explicit PitchedMemoryIterator(cudaPitchedPtr pitchedPtr, cudaExtent extent, cudaPos offset = make_cudaPos(0, 0, 0))
		{
			pitchedPtr_.pitch = pitchedPtr.pitch;
			pitchedPtr_.ptr = reinterpret_cast<unsigned char*>(pitchedPtr.ptr) + (offset.z*pitchedPtr.ysize + offset.y)*pitchedPtr.pitch + offset.x * sizeof(T);
			pitchedPtr_.ysize = pitchedPtr.ysize;
			pitchedPtr_.xsize = pitchedPtr.ysize;

			extent_.width = extent.width * sizeof(T);
			extent_.height = extent.height;
			extent_.depth = extent.depth;
		}

		__host__ __device__ __inline__
			void advance(difference_type n)
		{
			ptr_idx += n * sizeof(T);
		}
		__host__ __device__ __inline__
			void increment()
		{
			ptr_idx += sizeof(T);
		}
		__host__ __device__ __inline__
			void decrement()
		{
			ptr_idx -= sizeof(float);
		}
		__host__ __device__ __inline__
			bool equal(PitchedMemoryIterator const& other) const
		{
			return this->ptr_idx == other.ptr_idx;
		}
		/*__host__ */__device__ __inline__
			reference dereference() const
		{
			register int idx_x = ptr_idx % extent_.width;
			register int idx_y = (ptr_idx / extent_.width);
			register int idx_z = 0;
			if (extent_.depth > 1)
			{
				idx_y %= extent_.height;
				idx_z = ptr_idx / (extent_.height * extent_.width);
			}
			//printf("%g\n", *(T*)(reinterpret_cast<unsigned char*>(pitchedPtr_.ptr) + (idx_z*pitchedPtr_.ysize + idx_y)*pitchedPtr_.pitch + idx_x));
			//return *(reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(pitchedPtr_.ptr) + (ptr_idx / pitchedPtr_.xsize*pitchedPtr_.pitch + ptr_idx%pitchedPtr_.xsize)));
			return *(T*)(reinterpret_cast<unsigned char*>(pitchedPtr_.ptr)+(idx_z*pitchedPtr_.ysize + idx_y)*pitchedPtr_.pitch + idx_x);
			//return *(reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(pitchedPtr_.ptr) + pitchedPtr_.pitch*idx_y+ idx_x));
			//*(reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(pitchedPtr_.ptr) + (idx_z*pitchedPtr_.ysize + idx_y)*pitchedPtr_.pitch + idx_x));
		}
		__host__ __device__ __inline__
			difference_type distance_to(PitchedMemoryIterator const& other) const
		{
			return (other.ptr_idx - this->ptr_idx) / difference_type(sizeof(T));
			//return other.ptr_idx;
		}
	protected:
		cudaPitchedPtr pitchedPtr_;
		cudaExtent extent_;
		int ptr_idx = 0;
	};

}
