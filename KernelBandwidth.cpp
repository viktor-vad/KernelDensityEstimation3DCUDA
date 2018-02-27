#include "KernelBandwidth.h"
#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

KernelBandwidth::KernelBandwidth() :
	detH(1.0)
{
	H.m_12 = H.m_13 = H.m_23 = 0.0f;
	H.m_11 = H.m_22 = H.m_33 = 1.0;

	H_inv.m_12 = H_inv.m_13 = H_inv.m_23 = 0.0f;
	H_inv.m_11 = H_inv.m_22 = H_inv.m_33 = 1.0f;

	eig1 = eig2 = eig3 = 1.0;
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

	eig1 = eig2 = eig3 = s*s;
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

	detH = -H_33*H_12_2 + 2.0f * H_12*H_13*H_23 - H_22*H_13_2 - H_11*H_23_2 + H_11*H_22*H_33;

	H_inv.m_11 = (-H_23_2 + H_22*H_33) / detH;
	H_inv.m_12 = (H_13*H_23 - H_12*H_33) / detH;
	H_inv.m_13 = (H_12*H_23 - H_13*H_22) / detH;
	H_inv.m_22 = (H_11*H_33 - H_13_2) / detH;
	H_inv.m_23 = (H_12*H_13 - H_11*H_23) / detH;
	H_inv.m_33 = (H_11*H_22 - H_12_2) / detH;

	//eigen values;
	register float p1 = H_12_2 + H_13_2 + H_23_2;
	register float traceH = H_11 + H_22 + H_33;
	if (p1 <= FLT_EPSILON)
	{
		eig1 = fmaxf(fmaxf(H_11, H_22), H_33);
		eig3 = fminf(fminf(H_11, H_22), H_33);
	}
	else
	{
		float q = traceH / 3.0f;
		float p2 = 2.0f*p1;
		register float tmp = (H_11 - q);
		p2 += tmp*tmp;
		tmp = (H_22 - q);
		p2 += tmp*tmp;
		tmp = (H_33 - q);
		p2 += tmp*tmp;
		float p = sqrtf(p2 / 6.0f);
		float B_11 = H_11 - q;
		float B_22 = H_22 - q;
		float B_33 = H_33 - q;

		float detB= -B_33*H_12_2 + 2.0f * H_12*H_13*H_23 - B_22*H_13_2 - B_11*H_23_2 + B_11*B_22*B_33;
		float r = detB / (2.0f * p*p*p);

		float phi = acosf(r) / 3.0f;
		if (r >= 1.0f) phi = 0.0f;
		if (r <= -1.0f) phi = M_PI / 3.0f;

		eig1 = q + 2.0f * p * cos(phi);
		eig3 = q + 2.0f * p * cos(phi + (M_PI*2.0f / 3.0f));
	}
	eig2 = traceH - eig1 - eig3;

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

const float* KernelBandwidth::getMatrixDataPtr() const
{
	return &H.m_11;
}

float KernelBandwidth::getLargestEigenValue() const
{
	return eig1;
}

float KernelBandwidth::getMiddleEigenValue() const
{
	return eig2;
}

float KernelBandwidth::getSmallestEigenValue() const
{
	return eig3;
}

void KernelBandwidth::getEigenSystem(float & e1, float & e2, float & e3, float3 & eigv1, float3 & eigv2, float3 & eigv3) const
{
	e1 = eig1;
	e2 = eig2;
	e3 = eig3;
	computeEigenVectors(eigv1, eigv2, eigv3);
}

void KernelBandwidth::computeEigenVectors(float3& eigv1, float3& eigv2, float3& eigv3) const
{
	const float& H_11 = H.m_11;
	const float& H_12 = H.m_12;
	const float& H_13 = H.m_13;
	const float& H_22 = H.m_22;
	const float& H_23 = H.m_23;
	const float& H_33 = H.m_33;

	float3 row0 = make_float3(H_11 - eig3, H_12, H_13);
	float3 row1 = make_float3(H_12, H_22 - eig3, H_23);
	float3 row2 = make_float3(H_13, H_23, H_33 - eig3);
	float3 r0xr1 = cross(row0, row1);
	float3 r0xr2 = cross(row0, row2);
	float3 r1xr2 = cross(row1, row2);

	float d0 = dot(r0xr1, r0xr1);
	float d1 = dot(r0xr2, r0xr2);
	float d2 = dot(r1xr2, r1xr2);
	float dmax = d0;
	int imax = 0;

	if (d1 > dmax) { dmax = d1; imax = 1; }
	if (d2 > dmax) { imax = 2; }
	if (imax == 0)
	{
		eigv3 = r0xr1 / sqrtf(d0);
	}
	else if (imax == 1)	{
		eigv3 = r0xr2 / sqrtf(d1);
	}
	else //imax==2
	{
		eigv3 = r1xr2 / sqrtf(d2);
	}

	float3 U, V;
	float invLength;
	if (fabsf(eigv3.x) > fabsf(eigv3.y))
	{

		invLength = 1 / sqrtf(eigv3.x*eigv3.x + eigv3.z*eigv3.z);
		U = make_float3(-eigv3.z*invLength, 0.0f, eigv3.x*invLength);
	}
	else
	{
		invLength = 1 / sqrtf(eigv3.y*eigv3.y + eigv3.z*eigv3.z);
		U = make_float3(0.0f, eigv3.z*invLength, -eigv3.y*invLength);
	}
	V = cross(eigv3, U);

	float3 AU = make_float3(H_11*U.x + H_12*U.y + H_13*U.z, H_12*U.x + H_22*U.y + H_23*U.z, H_13*U.x + H_23*U.y + H_33*U.z);
	float3 AV = make_float3(H_11*V.x + H_12*V.y + H_13*V.z, H_12*V.x + H_22*V.y + H_23*V.z, H_13*V.x + H_23*V.y + H_33*V.z);

	float m00 = dot(U, AU) - eig2, m01 = dot(U, AV), m11 = dot(V, AV) - eig2;
	float absM00 = fabsf(m00), absM01 = fabsf(m01), absM11 = fabsf(m11);
	if (absM00 > absM11)
	{
		float maxAbsComp = fmaxf(absM00, absM01);
		if (maxAbsComp > 0)
		{
			if (absM00 >= absM01)
			{
				m01 /= m00; m00 = 1.0f / sqrtf(1.0f + m01 * m01); m01 *= m00;
			}
			else
			{
				m00 /= m01; m01 = 1.0f / sqrtf(1.0f + m00 * m00); m00 *= m01;
			}
			eigv2 = m01 * U - m00 * V;
		}
		else
		{
			eigv2 = U;
		}
	}
	else
	{
		float maxAbsComp = fmaxf(absM11, absM01);
		if (maxAbsComp > 0)
		{
			if (absM11 >= absM01)
			{
				m01 /= m11; m11 = 1.0f / sqrtf(1.0f + m01 * m01); m01 *= m11;
			}
			else
			{
				m11 /= m01; m01 = 1.0f / sqrtf(1.0f + m11 * m11); m11 *= m01;
			}
			eigv2 = m11 * U - m01 * V;
		}		else
		{
			eigv2 = U;
		}
	}

	eigv1 = cross(eigv3, eigv2);
}

void KernelBandwidth::setFromEigenSystem(const float & e1, const float & e2, const float & e3, const float3 & ev1, const float3 & ev2, const float3 & ev3)
{
	H.m_11= e1*ev1.x*ev1.x + e2*ev2.x*ev2.x + e3*ev3.x*ev3.x;
	H.m_12 = e1*ev1.x*ev1.y + e2*ev2.x*ev2.y + e3*ev3.x*ev3.y;
	H.m_13 = e1*ev1.x*ev1.z + e2*ev2.x*ev2.z + e3*ev3.x*ev3.z;
	H.m_22 = e1*ev1.y*ev1.y + e2*ev2.y*ev2.y + e3*ev3.y*ev3.y;
	H.m_23 = e1*ev1.y*ev1.z + e2*ev2.y*ev2.z + e3*ev3.y*ev3.z;
	H.m_33 = e1*ev1.z*ev1.z + e2*ev2.z*ev2.z + e3*ev3.z*ev3.z;

	eig1= e1;
	eig2= e2;
	eig3= e3;
	
	const float& H_11 = H.m_11;
	const float& H_12 = H.m_12;
	const float& H_13 = H.m_13;
	const float& H_22 = H.m_22;
	const float& H_23 = H.m_23;
	const float& H_33 = H.m_33;

	register float H_12_2 = H_12*H_12;
	register float H_13_2 = H_13*H_13;
	register float H_23_2 = H_23*H_23;

	detH = -H_33*H_12_2 + 2.0f * H_12*H_13*H_23 - H_22*H_13_2 - H_11*H_23_2 + H_11*H_22*H_33;

	H_inv.m_11 = (-H_23_2 + H_22*H_33) / detH;
	H_inv.m_12 = (H_13*H_23 - H_12*H_33) / detH;
	H_inv.m_13 = (H_12*H_23 - H_13*H_22) / detH;
	H_inv.m_22 = (H_11*H_33 - H_13_2) / detH;
	H_inv.m_23 = (H_12*H_13 - H_11*H_23) / detH;
	H_inv.m_33 = (H_11*H_22 - H_12_2) / detH;
}

void KernelBandwidth::getRotationOfEigenSystem(float& rot_x, float& rot_y, float& rot_z) const
{
	float3 eigv1, eigv2, eigv3;
	computeEigenVectors(eigv1, eigv2, eigv3);
	if (fabsf(eigv1.z) > std::numeric_limits<float>::epsilon())
	{
		rot_y = -asin(eigv1.z);
		float sign_cos_rot_y = 1.0f;
		if (cos(rot_y) < 0) sign_cos_rot_y = -1.0f;
		rot_x = atan2(eigv2.z *sign_cos_rot_y, eigv3.z *sign_cos_rot_y);
		rot_z = atan2(eigv1.y *sign_cos_rot_y, eigv1.x *sign_cos_rot_y);
	}
	else
	{
		rot_z = 0;
		if (eigv1.z < 0.0f)
		{
			rot_y = M_PI_2;
			rot_x = atan2f(eigv2.x, eigv3.x);
		}
		else
		{
			rot_y = -M_PI_2;
			rot_x = atan2f(-eigv2.x, -eigv3.x);
		}
	}
}

void KernelBandwidth::setFromEigenSystem(const float & e1, const float & e2, const float & e3, const float & rot_x, const float & rot_y, const float & rot_z)
{
	float3 eigv1, eigv2, eigv3;
	eigv1.x = cos(rot_y)*cos(rot_z);
	eigv1.y = cos(rot_y)*sin(rot_z);
	eigv1.z = -sin(rot_y);

	eigv2.x = sin(rot_x)*sin(rot_y)*cos(rot_z) - cos(rot_x)*sin(rot_z);
	eigv2.y = sin(rot_x)*sin(rot_y)*sin(rot_z) + cos(rot_x)*cos(rot_z);
	eigv2.z = sin(rot_x)*cos(rot_y);

	eigv3.x = cos(rot_x)*sin(rot_y)*cos(rot_z) + sin(rot_x)*sin(rot_z);
	eigv3.y = cos(rot_x)*sin(rot_y)*sin(rot_z) - sin(rot_x)*cos(rot_z);
	eigv3.z = cos(rot_x)*cos(rot_y);

	setFromEigenSystem(e1, e2, e3, eigv1, eigv2, eigv3);
}

bool  KernelBandwidth::isPositiveDefinite() const
{
	return (detH > 0.0f) && ((H.m_11*H.m_22 - H.m_12*H.m_12) > 0.0f);
}
//void KernelBandwidth::checkAndCorrectPositiveDefiniteness()
//{
//	if ((detH <= 0.0f) || (H.m_11*H.m_22 - H.m_12*H.m_12 <= 0.0f))
//	{
//		gte::SymmetricEigensolver3x3<float> solver;
//		std::array<float, 3> eval;
//		std::array<std::array<float, 3>, 3> evec;
//		solver(H.m_11, H.m_12, H.m_13, H.m_22, H.m_23, H.m_11, false, 0, eval, evec);
//		for (int i = 0; i<3; ++i) if (eval[i] <= 0.0f) eval[i] *= -1.0f;
//		SymmetricMatrix H_;
//		memset(&H_, 0, sizeof(H_));
//		for (int i = 0; i < 3; ++i)
//		{
//			H_.m_11 += eval[i] * evec[i][0] * evec[i][0];
//			H_.m_12 += eval[i] * evec[i][1] * evec[i][0];
//			H_.m_13 += eval[i] * evec[i][2] * evec[i][0];
//			H_.m_22 += eval[i] * evec[i][1] * evec[i][1];
//			H_.m_23 += eval[i] * evec[i][2] * evec[i][1];
//			H_.m_33 += eval[i] * evec[i][2] * evec[i][2];
//		}
//		setBandwidthMatrix(H_);
//	}
//}