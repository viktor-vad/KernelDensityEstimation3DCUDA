#pragma once
#include <cuda_runtime.h>
#include <helper_math.h>

struct SymmetricMatrix
{
	float m_11, m_12, m_13, m_22, m_23, m_33;
	__host__ __device__ float3 mult(float3 pos) const
	{
		return  make_float3(m_11, m_12, m_13)*pos.x + make_float3(m_12, m_22, m_23)*pos.y + make_float3(m_13, m_23, m_33)*pos.z;
	}
};

class KernelBandwidth
{
public:
	KernelBandwidth();
	~KernelBandwidth();
	void setIsotropic(const float& s);
	void setBandwidthMatrix(const float& H_11, const float& H_12, const float& H_13, const float& H_22, const float& H_23, const float& H_33);
	void setBandwidthMatrix(const SymmetricMatrix& other);
	void getInverseCoeffs(float& H_inv_11, float& H_inv_12, float& H_inv_13, float& H_inv_22, float& H_inv_23, float& H_inv_33) const;
	void getCoeffs(float& H_inv_11, float& H_inv_12, float& H_inv_13, float& H_inv_22, float& H_inv_23, float& H_inv_33) const;
	const SymmetricMatrix& getInverseMatrix() const;
	const SymmetricMatrix& getMatrix() const;
	SymmetricMatrix& getMatrixDataRef();
	float getDeterminant() const;
	float getLargestEigenValue() const;
	float getSmallestEigenValue() const;
	void checkAndCorrectPositiveDefiniteness();
private:
	SymmetricMatrix H;
	SymmetricMatrix H_inv;

	float detH;
};

