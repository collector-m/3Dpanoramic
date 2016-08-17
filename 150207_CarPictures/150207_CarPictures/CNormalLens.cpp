/*
	2015/02/07: ����ӽǱ任������(2015/02/08�����MatѰַ������)
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
	int Nx = src.x;	//����������
	int Ny = src.y;
	int Nz = 1;

	Mat N_src = (Mat_<float>(3,1)<<
		Nx,		Ny,		Nz);

	thetax = thetax/180*PI;	//�Ƕ�תΪ����
	thetay = thetay/180*PI;
	thetaz = thetaz/180*PI;

	float sf = 400;
	Mat T = (Mat_<float>(3,3)<<	//ƽ�ƾ���
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

	R = rotx*roty*rotz;		//��ת����
	Mat H = A1*R*T*A2;		//�任����

	Mat N_dst = H*N_src;		//�õ��任�������

	Point dst;
	dst.x = N_dst.at<float>(0,0)/N_dst.at<float>(2,0);	//���ر任�������
	dst.y = N_dst.at<float>(1,0)/N_dst.at<float>(2,0);

	return dst;
}

void CNormalLens::PerspectiveImage(Mat &src, Mat &dst, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale)
{
	thetax = thetax/180*PI;	//�Ƕ�תΪ����
	thetay = thetay/180*PI;
	thetaz = thetaz/180*PI;

	float sf = 400;
	Mat T = (Mat_<float>(3,3)<<	//ƽ�ƾ���
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

	R = rotx*roty*rotz;		//��ת����
	Mat H = A1*R*T*A2;		//�任����

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
	size_t memSize_src = height * src.step;	//��ȡͼƬ��С

	Size size_dst = Size((int)(width), (int)(height));	//����Ŀ��ͼ���С
	dst = Mat(size_dst, CV_64FC3);
	resize(src, dst, size_dst);

	int height_dst = dst.rows;	
	int width_dst = dst.cols;
	size_t memSize_dst = height_dst * dst.step;	//��ȡͼƬ��С

	uchar3* cuda_src = NULL;
	uchar3* cuda_dst = NULL;
	float* cuda_H = NULL;
	cudaMalloc((void**)&cuda_src, memSize_src);		//ΪGPU�����ڴ�
	cudaMalloc((void**)&cuda_dst, memSize_dst);
	cudaMalloc((void**)&cuda_H, 9*sizeof(float));

	cudaMemcpy(cuda_src, src.data, memSize_src, cudaMemcpyHostToDevice);	//����������GPU�Դ�
	cudaMemcpy(cuda_H, Ht, 9*sizeof(float), cudaMemcpyHostToDevice);

	cuda_Perspective_Caller(cuda_src, cuda_dst, height, width, height_dst, width_dst, cuda_H, scale);

	cudaMemcpy(dst.data, cuda_dst, memSize_dst, cudaMemcpyDeviceToHost);

	cudaFree(cuda_src);		//�ͷ�GPU�Դ�
	cudaFree(cuda_dst);
	cudaFree(cuda_H);
}