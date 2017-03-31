#pragma once

#include "Volume.h"


class VolumeBinningEstimator
{
public:
	static void estimateBinnedVolume(Volume&,const float4* samples_device, int nsamples);
};