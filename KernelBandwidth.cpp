#include "KernelBandwidth.h"


KernelBandwidth::KernelBandwidth() :
	detH(1.0)
{
	H.m_12 = H.m_13 = H.m_23 = 0.0f;
	H.m_11 = H.m_22 = H.m_33 = 1.0;

	H_inv.m_12 = H_inv.m_13 = H_inv.m_23 = 0.0f;
	H_inv.m_11 = H_inv.m_22 = H_inv.m_33 = 1.0f;
}


KernelBandwidth::~KernelBandwidth()
{
}

void KernelBandwidth::setIsotropic(const float& s)
{
	H.m_12 = H.m_13 = H.m_23 = 0.0f;
	H.m_11 = H.m_22 = H.m_33 = s*s;

	H_inv.m_12 = H_inv.m_13 = H_inv.m_23 = 0.0f;
	H_inv.m_11 = H_inv.m_22 = H_inv.m_33 = 1.0f / H.m_11;

	detH = H.m_11*H.m_22*H.m_33;
}

void KernelBandwidth::setBandwidthMatrix(const float& H_11, const float& H_12, const float& H_13, const float& H_22, const float& H_23, const float& H_33)
{
	H.m_11 = H_11;
	H.m_12 = H_12;
	H.m_13 = H_13;
	H.m_22 = H_22;
	H.m_23 = H_23;
	H.m_33 = H_33;

	register float H_12_2 = H_12*H_12;
	register float H_13_2 = H_13*H_13;
	register float H_23_2 = H_23*H_23;

	detH = -H_33*H_12_2 + 2 * H_12*H_13*H_23 - H_22*H_13_2 - H_11*H_23_2 + H_11*H_22*H_33;

	H_inv.m_11 = (-H_23_2 + H_22*H_33) / detH;
	H_inv.m_12 = (H_13*H_23 - H_12*H_33) / detH;
	H_inv.m_13 = (H_12*H_23 - H_13*H_22) / detH;
	H_inv.m_22 = (H_11*H_33 - H_13_2) / detH;
	H_inv.m_23 = (H_12*H_13 - H_11*H_23) / detH;
	H_inv.m_33 = (H_11*H_22 - H_12_2) / detH;

}

void KernelBandwidth::setBandwidthMatrix(const SymmetricMatrix& other)
{
	this->setBandwidthMatrix(other.m_11, other.m_12, other.m_13, other.m_22, other.m_23, other.m_33);
}

float  KernelBandwidth::getDeterminant() const
{
	return detH;
}

void KernelBandwidth::getInverseCoeffs(float& H_inv_11, float& H_inv_12, float& H_inv_13, float& H_inv_22, float& H_inv_23, float& H_inv_33) const
{
	H_inv_11 = this->H_inv.m_11;
	H_inv_12 = this->H_inv.m_12;
	H_inv_13 = this->H_inv.m_13;
	H_inv_22 = this->H_inv.m_22;
	H_inv_23 = this->H_inv.m_23;
	H_inv_33 = this->H_inv.m_33;
}

void KernelBandwidth::getCoeffs(float& H_11, float& H_12, float& H_13, float& H_22, float& H_23, float& H_33) const
{
	H_11 = this->H.m_11;
	H_12 = this->H.m_12;
	H_13 = this->H.m_13;
	H_22 = this->H.m_22;
	H_23 = this->H.m_23;
	H_33 = this->H.m_33;
}

const SymmetricMatrix& KernelBandwidth::getInverseMatrix() const
{
	return H_inv;
}

const SymmetricMatrix& KernelBandwidth::getMatrix() const
{
	return H;
}


SymmetricMatrix& KernelBandwidth::getMatrixDataRef()
{
	return H;
}

#include "3rdParty/GteSymmetricEigensolver3x3.h"

float KernelBandwidth::getLargestEigenValue() const
{
	gte::SymmetricEigensolver3x3<float> solver;
	std::array<float, 3> eval;
	std::array<std::array<float, 3>, 3> evec;
	solver(H.m_11, H.m_12, H.m_13, H.m_22, H.m_23, H.m_11, false, -1, eval, evec);
	return eval[0];
}

float KernelBandwidth::getSmallestEigenValue() const
{
	gte::SymmetricEigensolver3x3<float> solver;
	std::array<float, 3> eval;
	std::array<std::array<float, 3>, 3> evec;
	solver(H.m_11, H.m_12, H.m_13, H.m_22, H.m_23, H.m_11, false, -1, eval, evec);
	return eval[2];
}

void KernelBandwidth::checkAndCorrectPositiveDefiniteness()
{
	if ((detH <= 0.0f) || (H.m_11*H.m_22 - H.m_12*H.m_12 <= 0.0f))
	{
		gte::SymmetricEigensolver3x3<float> solver;
		std::array<float, 3> eval;
		std::array<std::array<float, 3>, 3> evec;
		solver(H.m_11, H.m_12, H.m_13, H.m_22, H.m_23, H.m_11, false, 0, eval, evec);
		for (int i = 0; i<3; ++i) if (eval[i] <= 0.0f) eval[i] *= -1.0f;
		SymmetricMatrix H_;
		memset(&H_, 0, sizeof(H_));
		for (int i = 0; i < 3; ++i)
		{
			H_.m_11 += eval[i] * evec[i][0] * evec[i][0];
			H_.m_12 += eval[i] * evec[i][1] * evec[i][0];
			H_.m_13 += eval[i] * evec[i][2] * evec[i][0];
			H_.m_22 += eval[i] * evec[i][1] * evec[i][1];
			H_.m_23 += eval[i] * evec[i][2] * evec[i][1];
			H_.m_33 += eval[i] * evec[i][2] * evec[i][2];
		}
		setBandwidthMatrix(H_);
	}
}