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
	const float* getMatrixDataPtr() const;
	float getDeterminant() const;
	float getLargestEigenValue() const;
	float getMiddleEigenValue() const;
	float getSmallestEigenValue() const;

	void getEigenSystem(float& e1, float& e2, float& e3, float3& ev1, float3& ev2, float3& ev3) const;
	void setFromEigenSystem(const float& e1, const float& e2, const float& e3, const float3& ev1, const float3& ev2, const float3& ev3);
	void getRotationOfEigenSystem(float& rot_x, float& rot_y, float& rot_z) const;
	void setFromEigenSystem(const float& e1, const float& e2, const float& e3, const float& rot_x, const float& rot_y, const float& rot_z);
	bool isPositiveDefinite() const;
private:
	SymmetricMatrix H;
	SymmetricMatrix H_inv;

	float detH;

	float eig1, eig2, eig3;
	//float3 eigv1, eigv2, eigv3;
	void computeEigenVectors(float3& eigv1, float3& eigv2, float3& eigv3) const;
};

