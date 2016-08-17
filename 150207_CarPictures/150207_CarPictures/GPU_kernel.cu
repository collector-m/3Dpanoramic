

#include "GPU_kernel.h"

#include <math.h>

#define CMV_MAX_BUF 1024
#define MAX_POL_LENGTH 64
#define M_PI 3.1415926


struct ocam_model
{
	double pol[MAX_POL_LENGTH];    // the polynomial coefficients: pol[0] + x"pol[1] + x^2*pol[2] + ... + x^(N-1)*pol[N-1]
	int length_pol;                // length of polynomial
	double invpol[MAX_POL_LENGTH]; // the coefficients of the inverse polynomial
	int length_invpol;             // length of inverse polynomial
	double xc;         // row coordinate of the center
	double yc;         // column coordinate of the center
	double c;          // affine parameter
	double d;          // affine parameter
	double e;          // affine parameter
	int width;         // image width
	int height;        // image height
};

__global__ void cuda_Perspective_kernel(const uchar3 *src, uchar3 *dst, int height, int width, int height_dst, int width_dst, float* H, float scale)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	float Nx = (i-height_dst/2.0)/scale;			//视角变换算法
	float Ny = (j-width_dst/2.0)/scale;
	float zz = H[6]*Nx + H[7]*Ny + H[8];

	int ii = (H[0]*Nx + H[1]*Ny + H[2])/zz + height/2;
	int jj = (H[3]*Nx + H[4]*Ny + H[5])/zz + width/2;

	if (i < height_dst && j < width_dst)
	{
		int offset = i * width_dst + j;
		if (jj > -1 && jj < width && ii < height && ii > -1)
		{
			int offset0 = ii * width +jj;
			uchar3 temp = src[offset0];
			dst[offset].x = temp.x;
			dst[offset].y = temp.y;
			dst[offset].z = temp.z;
		} 
		else
		{
			dst[offset].x = 0;
			dst[offset].y = 0;
			dst[offset].z = 0;
		}
	} 
}

void cuda_Perspective_Caller(const uchar3 *src, uchar3 *dst, int height, int width, int height_dst, int width_dst, float H[9], float scale)
{
	dim3 threads(16, 16);
	dim3 grids((width_dst + threads.x - 1)/threads.x, (height_dst + threads.y - 1)/threads.y);

	cuda_Perspective_kernel<<<grids, threads>>>(src, dst, height, width, height_dst, width_dst, H, scale);

	cudaThreadSynchronize();
}

__global__ void cuda_Undistort_kernel(float1* mapx, float1* mapy, struct ocam_model *m_model, int d_width, int d_height, float Nxc, float Nyc, float Nz, float xshift, float yshift, float sf, float scale)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	double *invpol     = m_model->invpol; 
	double xc          = (m_model->xc);
	double yc          = (m_model->yc); 
	double c           = (m_model->c);
	double d           = (m_model->d);
	double e           = (m_model->e);
	int    width       = (m_model->width);
	int    height      = (m_model->height);
	int length_invpol  = (m_model->length_invpol);

	double M[3];
	double m[2];

	M[0] = (i-Nxc-xshift)*scale;
	M[1] = (j-Nyc-yshift)*scale;
	M[2] = 1;

	double norm = sqrt(M[0]*M[0]+M[1]*M[1]);
	double theta = atan(-M[2]/norm*width/sf);

	double t_i, rho, x, y;
	if (norm != 0)
	{
		double invnorm = 1 / norm;
		rho = invpol[0];
		t_i = 1;

		for (int k = 1; k < length_invpol; k++)
		{
			t_i *= theta;
			rho += t_i*invpol[k];
		}

		x = M[0] * invnorm * rho;
		y = M[1] * invnorm * rho;

		m[0] = x*c + y*d + xc;
		m[1] = x*e + y + yc;
	}
	else
	{
		m[0] = xc;
		m[1] = yc;
	}

	if (i < d_height && j < d_width)
	{
		int offset = i * d_width + j;
		mapx[offset].x = m[1];
		mapy[offset].x = m[0];
	}
}

void cuda_Undistort_Caller(float1* mapx, float1* mapy, struct ocam_model *m_model, int d_width, int d_height, float Nxc, float Nyc, float Nz, float xshift, float yshift, float sf, float scale)
{
	// 分配线程块以及线程
	dim3 threads(32, 16);
	dim3 grids((d_width + threads.x - 1) / threads.x, (d_height + threads.y - 1) / threads.y);

	// 调用核函数
	cuda_Undistort_kernel<<<grids, threads >>>(mapx, mapy, m_model, d_width, d_height, Nxc, Nyc, Nz, xshift, yshift, sf, scale);

	// 线程同步
	cudaThreadSynchronize();
}

__global__ void cuda_UndistortPerspective_kernel(float1* mapx, float1* mapy, struct ocam_model *m_model, int d_width, int d_height, float Nxc, float Nyc, float Nz, float* H, float xshift, float yshift, float sf, float scale)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	double *invpol     = m_model->invpol; 
	double xc          = (m_model->xc);
	double yc          = (m_model->yc); 
	double c           = (m_model->c);
	double d           = (m_model->d);
	double e           = (m_model->e);
	int    width       = (m_model->width);
	int    height      = (m_model->height);
	int length_invpol  = (m_model->length_invpol);

	double M[3];
	double M1[3];
	double m[2];

	M1[0] = (i-Nxc-xshift)/Nz*scale;
	M1[1] = (j-Nyc-yshift)/Nz*scale;
	M1[2] = 1;

	M[0] = H[0]*M1[0]+H[1]*M1[1]+H[2];
	M[1] = H[3]*M1[0]+H[4]*M1[1]+H[5];
	M[2] = H[6]*M1[0]+H[7]*M1[1]+H[8];

	double norm = sqrt(M[0]*M[0]+M[1]*M[1]);
	double theta = atan(-M[2]/norm*width/sf);

	double t_i, rho, x, y;
	if (norm != 0)
	{
		double invnorm = 1 / norm;
		rho = invpol[0];
		t_i = 1;

		for (int k = 1; k < length_invpol; k++)
		{
			t_i *= theta;
			rho += t_i*invpol[k];
		}

		x = M[0] * invnorm * rho;
		y = M[1] * invnorm * rho;

		m[0] = x*c + y*d + xc;
		m[1] = x*e + y + yc;
	}
	else
	{
		m[0] = xc;
		m[1] = yc;
	}

	if (i < d_height && j < d_width)
	{
		int offset = i * d_width + j;
		mapx[offset].x = m[1];
		mapy[offset].x = m[0];
	}
}

void cuda_UndistortPerspective_Caller(float1* mapx, float1* mapy, struct ocam_model *m_model, int d_width, int d_height, float Nxc, float Nyc, float Nz, float* H, float xshift, float yshift, float sf, float scale)
{
	// 分配线程块以及线程
	dim3 threads(32, 16);
	dim3 grids((d_width + threads.x - 1) / threads.x, (d_height + threads.y - 1) / threads.y);

	// 调用核函数
	cuda_UndistortPerspective_kernel<<<grids, threads >>>(mapx, mapy, m_model, d_width, d_height, Nxc, Nyc, Nz, H, xshift, yshift, sf, scale);

	// 线程同步
	cudaThreadSynchronize();
}

//__global__ void cuda_SurroundView_kernel(const uchar3 *front,const float1 *front_mapx,const float1 *front_mapy,
//	const uchar3 *frontleft,const float1 *frontleft_mapx,const float1 *frontleft_mapy,
//	const uchar3 *frontright,const float1 *frontright_mapx,const float1 *frontright_mapy,
//	const uchar3 *rear,const float1 *rear_mapx,const float1 *rear_mapy,
//	const uchar3 *rearleft,const float1 *rearleft_mapx,const float1 *rearleft_mapy,
//	const uchar3 *rearright,const float1 *rearright_mapx,const float1 *rearright_mapy,
//	const float1 *frontleft_coeff,const float1 *frontright_coeff,const float1 *left_coeff,const float1 *right_coeff,const float1 *rearleft_coeff,const float1 *rearright_coeff,
//	int height, int width,
//	uchar3 *dst)
//{
//	int j = threadIdx.x + blockIdx.x * blockDim.x;
//	int i = threadIdx.y + blockIdx.y * blockDim.y;
//
//
//}
//
////计算环视图像回调函数
//void cuda_SurroundView_Caller(const uchar3 *front,const float1 *front_mapx,const float1 *front_mapy,
//	const uchar3 *frontleft,const float1 *frontleft_mapx,const float1 *frontleft_mapy,
//	const uchar3 *frontright,const float1 *frontright_mapx,const float1 *frontright_mapy,
//	const uchar3 *rear,const float1 *rear_mapx,const float1 *rear_mapy,
//	const uchar3 *rearleft,const float1 *rearleft_mapx,const float1 *rearleft_mapy,
//	const uchar3 *rearright,const float1 *rearright_mapx,const float1 *rearright_mapy,
//	const float1 *frontleft_coeff,const float1 *frontright_coeff,const float1 *left_coeff,const float1 *right_coeff,const float1 *rearleft_coeff,const float1 *rearright_coeff,
//	int height, int width,
//	uchar3 *dst)
//{
//	// 分配线程块以及线程
//	dim3 threads(32, 16);
//	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
//
//	// 调用核函数
//	cuda_SurroundView_kernel<<<grids, threads >>>(front,front_mapx,front_mapy,
//		frontleft,frontleft_mapx,frontleft_mapy,
//		frontright,frontright_mapx,frontright_mapy,
//		rear,rear_mapx,rear_mapy,
//		rearleft,rearleft_mapx,rearleft_mapy,
//		rearright,rearright_mapx,rearright_mapy,
//		frontleft_coeff,frontright_coeff,left_coeff,right_coeff,rearleft_coeff,rearright_coeff,
//		height, width,
//		dst);
//
//	// 线程同步
//	cudaThreadSynchronize();
//}

__global__ void cuda_imageMulcoeff_kernel(const float3 *src, const float1 *coeff, float3 *dst, int width, int height)
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = i*width+j;
	//float temp_x, temp_y, temp_z, temp_coeff;
	//temp_x = src[offset].x;
	//temp_y = src[offset].y;
	//temp_z = src[offset].z;
	//temp_coeff = coeff[offset].x;

	dst[offset].x = src[offset].x*coeff[offset].x;
	dst[offset].y = src[offset].y*coeff[offset].x;
	dst[offset].z = src[offset].z*coeff[offset].x;
}

void cuda_imageMulcoeff_Caller(const float3 *src, const float1 *coeff, float3 *dst, int width, int height)
{
	// 分配线程块以及线程
	dim3 threads(32, 16);
	dim3 grids((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

	// 调用核函数
	cuda_imageMulcoeff_kernel<<<grids, threads >>>(src, coeff, dst, width, height);

	// 线程同步
	cudaThreadSynchronize();
}