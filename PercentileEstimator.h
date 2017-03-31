#pragma once
class Volume;

#include <vector>

class PercentileEstimator
{
public:
	static void estimatePercentileIsovalues(const Volume& volume, const std::vector<float>& percentiles, std::vector<float>& isovalues, const float& scale_factor);
	static void estimatePercentileIsovalues(const Volume& volume, const float* percentiles, const int percentile_num,float* isovalues, const float& scale_factor);
};