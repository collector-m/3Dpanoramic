/*
	2015/02/07: 点的视角变换有问题(2015/02/08解决：Mat寻址的问题)
*/

#include "CNormalLens.h"
#include "GPU_kernel.h"

#include <math.h>

#define PI 3.14159

CNormalLens::CNormalLens()
{

}

CNormalLens::~CNormalLens()
{

}

Point CNormalLens::PerspectivePoint(Point src, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale)
{
	int Nx = src.x;	//输入点的坐标
	int Ny = src.y;
	int Nz = 1;

	Mat N_src = (Mat_<float>(3,1)<<
		Nx,		Ny,		Nz);

	thetax = thetax/180*PI;	//角度转为弧度
	thetay = thetay/180*PI;
	thetaz = thetaz/180*PI;

	float sf = 400;
	Mat T = (Mat_<float>(3,3)<<	//平移矩阵
		1,		0,	xshift/sf,
		0,		1,	yshift/sf,
		0,		0,		1);
	Mat A1 = (Mat_<float>(3,3)<<	
		sf,		0,		0,
		0,		sf,		0,
		0,		0,		1);
	Mat A2 = (Mat_<float>(3,3)<<	
		1/sf,	0,		0,
		0,		1/sf,	0,
		0,		0,		1);

	Mat R, rotx, roty, rotz;

	rotx = (Mat_<float>(3,3)<<
		1,		0,				0,
		0,		cos(thetax),	-sin(thetax),
		0,		sin(thetax),	cos(thetax));
	roty = (Mat_<float>(3,3)<<
		cos(thetay),	0,	sin(thetay),
		0,				1,		0,
		-sin(thetay),	0,		cos(thetay));
	rotz = (Mat_<float>(3,3)<<
		cos(thetaz),	-sin(thetaz),	0,
		sin(thetaz),	cos(thetaz),	0,
		0,				0,				1);

	R = rotx*roty*rotz;		//旋转矩阵
	Mat H = A1*R*T*A2;		//变换矩阵

	Mat N_dst = H*N_src;		//得到变换后的坐标

	Point dst;
	dst.x = N_dst.at<float>(0,0)/N_dst.at<float>(2,0);	//返回变换后的坐标
	dst.y = N_dst.at<float>(1,0)/N_dst.at<float>(2,0);

	return dst;
}

void CNormalLens::PerspectiveImage(Mat &src, Mat &dst, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale)
{
	thetax = thetax/180*PI;	//角度转为弧度
	thetay = thetay/180*PI;
	thetaz = thetaz/180*PI;

	float sf = 400;
	Mat T = (Mat_<float>(3,3)<<	//平移矩阵
		1,		0,	xshift/sf,
		0,		1,	yshift/sf,
		0,		0,		1);
	Mat A1 = (Mat_<float>(3,3)<<	
		sf,		0,		0,
		0,		sf,		0,
		0,		0,		1);
	Mat A2 = (Mat_<float>(3,3)<<	
		1/sf,	0,		0,
		0,		1/sf,	0,
		0,		0,		1);

	Mat R, rotx, roty, rotz;

	rotx = (Mat_<float>(3,3)<<
		1,		0,				0,
		0,		cos(thetax),	-sin(thetax),
		0,		sin(thetax),	cos(thetax));
	roty = (Mat_<float>(3,3)<<
		cos(thetay),	0,	sin(thetay),
		0,				1,		0,
		-sin(thetay),	0,	cos(thetay));
	rotz = (Mat_<float>(3,3)<<
		cos(thetaz),	-sin(thetaz),	0,
		sin(thetaz),	cos(thetaz),	0,
		0,				0,				1);

	R = rotx*roty*rotz;		//旋转矩阵
	Mat H = A1*R*T*A2;		//变换矩阵

	float Ht[9] = {0};
	for (int i = 0; i<3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Ht[i*3+j] = H.at<float>(i,j);
		}
	}


	int height = src.rows;	
	int width = src.cols;
	size_t memSize_src = height * src.step;	//获取图片大小

	Size size_dst = Size((int)(width), (int)(height));	//分配目标图像大小
	dst = Mat(size_dst, CV_64FC3);
	resize(src, dst, size_dst);

	int height_dst = dst.rows;	
	int width_dst = dst.cols;
	size_t memSize_dst = height_dst * dst.step;	//获取图片大小

	uchar3* cuda_src = NULL;
	uchar3* cuda_dst = NULL;
	float* cuda_H = NULL;
	cudaMalloc((void**)&cuda_src, memSize_src);		//为GPU创建内存
	cudaMalloc((void**)&cuda_dst, memSize_dst);
	cudaMalloc((void**)&cuda_H, 9*sizeof(float));

	cudaMemcpy(cuda_src, src.data, memSize_src, cudaMemcpyHostToDevice);	//传递数据至GPU显存
	cudaMemcpy(cuda_H, Ht, 9*sizeof(float), cudaMemcpyHostToDevice);

	cuda_Perspective_Caller(cuda_src, cuda_dst, height, width, height_dst, width_dst, cuda_H, scale);

	cudaMemcpy(dst.data, cuda_dst, memSize_dst, cudaMemcpyDeviceToHost);

	cudaFree(cuda_src);		//释放GPU显存
	cudaFree(cuda_dst);
	cudaFree(cuda_H);
}