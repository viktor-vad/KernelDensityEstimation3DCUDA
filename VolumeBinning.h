#pragma once

#include "Volume.h"


class VolumeBinningEstimator
{
public:
	template <typename CONST_SAMPLES_IT_TYPE>
	static void estimateBinnedVolumeFromIterator(Volume&, CONST_SAMPLES_IT_TYPE samples_device, int nsamples, bool linear = false);

	static void estimateBinnedVolume(Volume& volume, const float4* __restrict__ samples_device, int nsamples, bool linear = false);

};