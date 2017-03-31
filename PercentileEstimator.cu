#include "PercentileEstimator.h"
#include <algorithm>

#include "Volume.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

void PercentileEstimator::estimatePercentileIsovalues(const Volume& volume, const std::vector<float>& percentiles, std::vector<float>& isovalues, const float& scale_factor)
{
	isovalues.resize(percentiles.size());
	estimatePercentileIsovalues(volume, percentiles.data(), percentiles.size(), isovalues.data(), scale_factor);
}

void PercentileEstimator::estimatePercentileIsovalues(const Volume& volume, const float* percentiles, const int percentile_num, float* isovalues, const float& scale_factor)
{
	if (!percentile_num) return;
	if (!percentiles) return;
	if (!isovalues) return;

	memset(isovalues, 0, sizeof(float)*percentile_num);

	thrust::device_vector<float> sorted_cells(volume.numberOfCells());
	volume.copyFlattened(thrust::raw_pointer_cast(sorted_cells.data()), cudaMemcpyDeviceToDevice);
	thrust::stable_sort(sorted_cells.begin(), sorted_cells.end(),thrust::greater<float>());

	float sum = 0.0f;
	std::vector<float> tmp(256);
	int last;
	int perc_i = 0;
	float val2find = percentiles[0] * scale_factor;
	for (int i = 0; i < sorted_cells.size(); i += 256)
	{
		last = std::min<int>(sorted_cells.size() - i, 256);
		thrust::copy_n(sorted_cells.begin() + i, last, tmp.begin());
		for (int ii = 0; ii < last; ++ii)
		{
			sum += tmp[ii];
			if (sum >= val2find) {
				isovalues[perc_i] = tmp[ii];
				perc_i++;
				if (perc_i == percentile_num) return;
				val2find = percentiles[perc_i] * scale_factor;
			}
		}
	}
}