
#include <stdlib.h>
#include <vector>
#include <numeric>

#include <csv.h>

#include <cuda_runtime.h>

#include <SilvermanBandwidthEstimator.h>
#include <KernelBandwidth.h>

#include <nmsimplex.hpp>
#include <KernelDensityEstimationCUDA.h>
#include <FFTLSCVEstimatorCUDA.h>
#include <VolumeBinning.h>
#include <Volume.h>

int main(int argc, char** argv)
{
	std::vector<float4> samples;
	io::CSVReader<3> in("molecule.csv");
	
	in.read_header(io::ignore_extra_column, "x", "y", "z");
	float x, y, z;
	while (in.read_row(x, y, z))
	{
		samples.push_back(make_float4(x, y, z, 1.0));
	}

	int samples_num = samples.size();

	thrust::device_vector<float4> samples_device(samples.begin(), samples.end());
	
	Volume* pVolume = new Volume();
	pVolume->setGridSize(make_uint3(128));
	
	BoundingBox BB = estimateBoundingBox(thrust::raw_pointer_cast(samples_device.data()), samples_device.size());
	pVolume->setBox(BB);
	pVolume->fillWithZeros();
	
	KernelBandwidth kernelBandwidth;
	SilvermanBandwidthEstimator::estimateBandwidth(thrust::raw_pointer_cast(samples_device.data()), samples_num, kernelBandwidth);
	
	VolumeBinningEstimator::estimateBinnedVolume(*pVolume, thrust::raw_pointer_cast(samples_device.data()), samples_num,true);

	float lscv = FFTLSCVEstimatorCUDA::estimateLSCV(*pVolume, samples_num, kernelBandwidth);
	printf("init lscv: %.15g\n", lscv);

	
	NMSimplexT<float> nmSimplex;
	std::vector<float> vechH(6);
	kernelBandwidth.getCoeffs(vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
	
	float detBoundary = kernelBandwidth.getDeterminant()*0.01f;
	float currentBest = lscv;
	printf("[%.15g %.15g %.15g %.15g %.15g %.15g] \n", vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
	nmSimplex.objfunc = [&](float* vechH)->float
	{
		kernelBandwidth.setBandwidthMatrix(vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
		
		float tmp= FFTLSCVEstimatorCUDA::estimateLSCV(*pVolume,samples_num,kernelBandwidth);
		float tmp2 = kernelBandwidth.getDeterminant();
		float tmp3 = vechH[0] * vechH[3] - vechH[1] * vechH[1];
		float tmp4 = kernelBandwidth.getSmallestEigenValue();

		if (tmp2 < detBoundary) tmp = FLT_MAX;
		if (tmp != tmp) tmp = FLT_MAX;

		printf("tmp %.15g tmp2 %.15g tmp3 %.15g tmp4 %.15g\n", tmp, tmp2, tmp3, tmp4);

		if (tmp < currentBest)
		{
			printf("[%.15g %.15g %.15g %.15g %.15g %.15g] \n", vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
			currentBest = tmp;
		}
		return tmp;
	};

	nmSimplex.EPSILON = FLT_EPSILON*lscv;
	nmSimplex.scale = sqrtf(kernelBandwidth.getSmallestEigenValue());
	nmSimplex.MAX_IT = 100;
	nmSimplex.Optimize(vechH.data(), 6);
	
	kernelBandwidth.setBandwidthMatrix(vechH[0], vechH[1], vechH[2], vechH[3], vechH[4], vechH[5]);
	

	float LSCV = FFTLSCVEstimatorCUDA::estimateLSCV(*pVolume, samples_num, kernelBandwidth);
	printf("LSCV: %.15g\n",LSCV);

	KernelDensityEstimationCUDA kdeCUDA;
	kdeCUDA.estimateBinned(*pVolume, kernelBandwidth);
	std::vector<float> debug3(pVolume->numberOfCells(), 0.0f);
	pVolume->copyFlattened(debug3.data());

	FILE * pFile;
	pFile = fopen("density.csv", "w");
	fprintf(pFile,"x,y,z,density\n");
	int idx = 0;
	for (int k = 0; k<pVolume->gridSize().z; ++k)
		for (int j = 0; j<pVolume->gridSize().y; ++j)
			for (int i = 0; i < pVolume->gridSize().x; ++i)
			{
				float3 cellPos = make_float3(float(i) + 0.5f, float(j) + 0.5f, float(k) + 0.5f)*pVolume->cellSize() + make_float3(BB.lower);
				fprintf(pFile,"%g,%g,%g,%g\n", cellPos.x, cellPos.y, cellPos.z, debug3[idx]);
				++idx;
			}
	fclose(pFile);

	delete pVolume;
	return EXIT_SUCCESS;
}
