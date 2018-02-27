
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <random>

#include <csv.h>

#include <NormalScaleBandwidthEstimator.h>
#include <KernelBandwidth.h>


#include <KernelDensityEstimationCUDA.h>
#include <FFTLSCVEstimatorCUDA.h>
#include <FFTKDECUDAMemoryManager.h>

#include <VolumeBinning.h>
#include <Volume.h>
#include <PercentileEstimator.h>

#include <LSCVOptimzer.h>

int main(int argc, char** argv)
{
	Volume* pVolume = new Volume();
	pVolume->setGridSize(128, 128, 128);

	bool optimizeBandwidth = true;

	std::vector<float4> samples;
	io::CSVReader<3> in("molecule.csv");
	
	in.read_header(io::ignore_extra_column, "x", "y", "z");
	
	float x, y, z;
	while (in.read_row(x, y, z))
	{
		samples.push_back(make_float4(x, y, z, 1.0));
	}
	std::size_t samples_num = samples.size();


	thrust::device_vector<float4> samples_device = samples;
	KernelBandwidth kernelBandwidth;
	NormalScaleBandwidthEstimator::estimateBandwidth(thrust::raw_pointer_cast(samples_device.data()), samples_num, kernelBandwidth);

	BoundingBox BB = estimateBoundingBox(thrust::raw_pointer_cast(samples_device.data()), samples_device.size());

	BB.lower -= kernelBandwidth.getLargestEigenValue();
	BB.upper += kernelBandwidth.getLargestEigenValue();

	pVolume->setBox(BB);

	pVolume->fillWithZeros();
	VolumeBinningEstimator::estimateBinnedVolume(*pVolume, thrust::raw_pointer_cast(samples_device.data()), samples_num, true);

	std::shared_ptr<FFTKDECUDAMemoryManager> memoryManagerPtr = FFTKDECUDAMemoryManager::New();

	if (optimizeBandwidth)
	{
		LSCVOptimizer lscvOptimizer(pVolume, samples_num);
		lscvOptimizer.getEstimator()->setMemoryManager(memoryManagerPtr);
		lscvOptimizer.Optimize(kernelBandwidth);

	}

	const SymmetricMatrix& matrix = kernelBandwidth.getMatrix();
	printf("[\n[%f %f %f]\n[%f %f %f]\n[%f %f %f]\n]\n", matrix.m_11, matrix.m_12, matrix.m_13, matrix.m_12, matrix.m_22, matrix.m_23, matrix.m_13, matrix.m_23, matrix.m_33);

	KernelDensityEstimationCUDA kdeCUDA;
	kdeCUDA.setMemoryManager(memoryManagerPtr);
	kdeCUDA.estimateBinned(*pVolume, kernelBandwidth);

	memoryManagerPtr->release();
	//
	printf("estimated\n");

	float scale_factor = float(samples_num) / (pVolume->cellSize().x*pVolume->cellSize().y*pVolume->cellSize().z);
	std::vector<float> percentile_buff = { 0.25f,0.5f,0.75f,0.9f };
	std::vector<float> isovalues_buff(4, 0.0f);

	PercentileEstimator::estimatePercentileIsovalues(*pVolume, percentile_buff, isovalues_buff, scale_factor);

	for (int i = 0; i < percentile_buff.size(); ++i)
	{
		printf("%d%% %.15g\n", int(percentile_buff[i] * 100), isovalues_buff[i]);
	}

	delete pVolume;


	return EXIT_SUCCESS;
}