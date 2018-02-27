#include "NormalScaleBandwidthEstimator.h"
#include <algorithm>

#include <helper_math.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "KernelBandwidth.h"


struct getXX_YY_ZZ
{
	getXX_YY_ZZ(float4 mean):mean_(mean)
	{}
	float4 mean_;
	__device__ __host__ float4 operator()(const float4 f4_ls)
	{
		float4 tmp = f4_ls - mean_;
		return tmp*tmp;
	}
};

struct getXY_YZ_XZ
{
	getXY_YZ_XZ(float4 mean):mean_(mean)
	{}
	float4 mean_;
	__device__ __host__ float4 operator()(const float4 f4)
	{
		float4 tmp = f4 - mean_;
		return tmp*make_float4(tmp.y,tmp.z,tmp.x,0.0f);
	}
};

void  NormalScaleBandwidthEstimator::estimateBandwidth(const float4* samples_device, const int sample_num, KernelBandwidth& kernelBandwidth)
{

	//thrust::device_vector<float4> samples_device(sample_num);
	//thrust::copy_n(thrust::device_pointer_cast<const float4>(samples_dev), sample_num, samples_device.begin());
	thrust::device_ptr<const float4> samples_device_begin(samples_device);
	thrust::device_ptr<const float4> samples_device_end(samples_device+sample_num);

	float4 sample_mean = thrust::reduce(samples_device_begin, samples_device_end,make_float4(0.0f))/float(sample_num);

	//thrust::host_vector<float4> debug(samples_device);

	//thrust::transform(samples_device_begin, samples_device_end,samples_device_begin, thrust::placeholders::_1 - sample_mean);
	//debug = samples_device;

	//register float coeff = 1.5393339588134 * ::powf(float(sample_num), -2.0f / 7.0f)/std::max<float>(1.0,sample_num-1);
	register float coeff = 2.0f*powf(2.0f / 3.0f / float(sample_num), 2.0f / 5.0f) / std::max<float>(1.0, sample_num - 1);
	thrust::device_vector<float4> temp_device(sample_num);
	thrust::transform(samples_device_begin, samples_device_end, temp_device.begin(), getXX_YY_ZZ(sample_mean));
	float4 tmp = thrust::reduce(temp_device.begin(), temp_device.end(), make_float4(0.0, 0.0, 0.0, 0.0)) *coeff;
	thrust::transform(samples_device_begin, samples_device_end, temp_device.begin(), getXY_YZ_XZ(sample_mean));
	float4 tmp2 = thrust::reduce(temp_device.begin(), temp_device.end(), make_float4(0.0, 0.0, 0.0, 0.0))*coeff;

	kernelBandwidth.setBandwidthMatrix(tmp.x, tmp2.x, tmp2.z, tmp.y, tmp2.y, tmp.z);

}


void  NormalScaleBandwidthEstimator::estimateBandwidthAA(const float4* samples_device, const int sample_num, KernelBandwidth& kernelBandwidth)
{
	thrust::device_ptr<const float4> samples_device_begin(samples_device);
	thrust::device_ptr<const float4> samples_device_end(samples_device + sample_num);

	float4 sample_mean = thrust::reduce(samples_device_begin, samples_device_end, make_float4(0.0f)) / float(sample_num);

	register float coeff = 2.0f*powf(2.0f / 3.0f / float(sample_num), 2.0f / 5.0f) / std::max<float>(1.0, sample_num - 1);
	thrust::device_vector<float4> temp_device(sample_num);
	thrust::transform(samples_device_begin, samples_device_end, temp_device.begin(), getXX_YY_ZZ(sample_mean));
	float4 tmp = thrust::reduce(temp_device.begin(), temp_device.end(), make_float4(0.0, 0.0, 0.0, 0.0)) *coeff;
	//thrust::transform(samples_device_begin, samples_device_end, temp_device.begin(), getXY_YZ_XZ(sample_mean));
	//float4 tmp2 = thrust::reduce(temp_device.begin(), temp_device.end(), make_float4(0.0, 0.0, 0.0, 0.0))*coeff;

	kernelBandwidth.setBandwidthMatrix(tmp.x, 0.0f, 0.0f, tmp.y, 0.0f, tmp.z);

}