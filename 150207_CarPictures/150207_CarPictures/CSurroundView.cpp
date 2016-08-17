

#include "CSurroundView.h"
#include "GPU_kernel.h"

#include <fstream>
#include <iostream>

using namespace std;

CSurroundView::CSurroundView()
{

}

CSurroundView::~CSurroundView()
{

}

bool CSurroundView::Get_ocammodel(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright)
{
	int i = -1;

	i = m_front.Initial(front);
	if (i != 0)
	{
		return false;
	}

	i = m_frontleft.Initial(frontleft);
	if (i != 0)
	{
		return false;
	}

	i = m_frontright.Initial(frontright);
	if (i != 0)
	{
		return false;
	}

	i = m_rear.Initial(rear);
	if (i != 0)
	{
		return false;
	}

	i = m_rearleft.Initial(rearleft);
	if (i != 0)
	{
		return false;
	}

	i = m_rearright.Initial(rearright);
	if (i != 0)
	{
		return false;
	}

	return true;
}

Mat readdata(char *filename, int width, int height)
{
	//int width = 0;
	//int height = 0;

	ifstream infile;
	infile.open(filename);

	//infile>>width;		//读取系数矩阵的长宽
	//infile>>height;
	//float temp;

	Mat result = Mat(Size(width, height), CV_32FC1);

	while (!infile.eof())
	{
		for (int j = 0; j<height; j++)
		{
			for(int i = 0; i<width; i++)
			{
				infile>>result.at<float>(j,i);
				//infile>>temp;
			}
		}
	}

	infile.close();

	return result;
}

void CSurroundView::Get_blendcoeff(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright)
{
	int height = 1024;			//系数矩阵的大小，与对应的视角变换后的图大小一致
	int width = 1280;
	m_front_coeff = readdata(front, width, height);	
	m_frontleft_coeff = readdata(frontleft, height, width);
	m_frontright_coeff = readdata(frontright, height, width);

	m_rear_coeff = readdata(rear, width, height);
	m_rearleft_coeff = readdata(rearleft, height, width);
	m_rearright_coeff = readdata(rearright, height, width);
}

void CSurroundView::Get_persparas(char *filename)
{
	ifstream infile(filename);

	if (!infile.is_open())
	{
		cout<<"Can't open file!"<<endl;
	}

	struct Per_Paras temp;
	char buf[128];
								/*2015/02/10上午11:00调试：scanf出现问题*/
	infile.getline(buf, 128);
	sscanf(buf, "%f %f %f %f %f %f %f", &(temp.thetax),&(temp.thetay),&(temp.thetaz),&(temp.xshift),&(temp.yshift),&(temp.sf),&(temp.scale));
	m_front_paras = temp;

	infile.getline(buf, 128);
	sscanf(buf, "%f %f %f %f %f %f %f", &(temp.thetax),&(temp.thetay),&(temp.thetaz),&(temp.xshift),&(temp.yshift),&(temp.sf),&(temp.scale));
	m_frontleft_paras = temp;

	infile.getline(buf, 128);
	sscanf(buf, "%f %f %f %f %f %f %f", &(temp.thetax),&(temp.thetay),&(temp.thetaz),&(temp.xshift),&(temp.yshift),&(temp.sf),&(temp.scale));
	m_frontright_paras = temp;

	infile.getline(buf, 128);
	sscanf(buf, "%f %f %f %f %f %f %f", &(temp.thetax),&(temp.thetay),&(temp.thetaz),&(temp.xshift),&(temp.yshift),&(temp.sf),&(temp.scale));
	m_rear_paras = temp;

	infile.getline(buf, 128);
	sscanf(buf, "%f %f %f %f %f %f %f", &(temp.thetax),&(temp.thetay),&(temp.thetaz),&(temp.xshift),&(temp.yshift),&(temp.sf),&(temp.scale));
	m_rearleft_paras = temp;

	infile.getline(buf, 128);
	sscanf(buf, "%f %f %f %f %f %f %f", &(temp.thetax),&(temp.thetay),&(temp.thetaz),&(temp.xshift),&(temp.yshift),&(temp.sf),&(temp.scale));
	m_rearright_paras = temp;
}

void CSurroundView::Create_6_UndistortPerspective_LUT()
{
	//前视摄像头创建LUT
	float thetax = m_front_paras.thetax;
	float thetay = m_front_paras.thetay;
	float thetaz = m_front_paras.thetaz;
	float xshift = m_front_paras.xshift+512;		//此处+512是为了和Matlab得到的参数一致
	float yshift = m_front_paras.yshift;
	float sf = m_front_paras.sf;
	float scale = m_front_paras.scale;
	m_front_mapx = Mat(Size(1280, 1024), CV_32FC1);		/*-------2015/02/10待处理:此处把结果图写死了，还未处理--------*/
	m_front_mapy = Mat(Size(1280, 1024), CV_32FC1);
	m_front.Create_UndistortPerspective_LUT(m_front_mapx,m_front_mapy,thetax,thetay,thetaz,xshift,yshift,sf,scale);
	//m_front.Create_UndistortPerspective_LUT(m_front_mapx,m_front_mapy,30,0,0,0,0,2,1);

	//前左视摄像头创建LUT
	thetax = m_frontleft_paras.thetax;
	thetay = m_frontleft_paras.thetay;
	thetaz = m_frontleft_paras.thetaz;
	xshift = m_frontleft_paras.xshift+512;
	yshift = m_frontleft_paras.yshift;
	sf = m_frontleft_paras.sf;
	scale = m_frontleft_paras.scale;
	m_frontleft_mapx = Mat(Size(1280, 1024), CV_32FC1);
	m_frontleft_mapy = Mat(Size(1280, 1024), CV_32FC1);
	m_frontleft.Create_UndistortPerspective_LUT(m_frontleft_mapx,m_frontleft_mapy,thetax,thetay,thetaz,xshift,yshift,sf,scale);

	//前右视摄像头创建LUT
	thetax = m_frontright_paras.thetax;
	thetay = m_frontright_paras.thetay;
	thetaz = m_frontright_paras.thetaz;
	xshift = m_frontright_paras.xshift+512;
	yshift = m_frontright_paras.yshift;
	sf = m_frontright_paras.sf;
	scale = m_frontright_paras.scale;
	m_frontright_mapx = Mat(Size(1280, 1024), CV_32FC1);
	m_frontright_mapy = Mat(Size(1280, 1024), CV_32FC1);
	m_frontright.Create_UndistortPerspective_LUT(m_frontright_mapx,m_frontright_mapy,thetax,thetay,thetaz,xshift,yshift,sf,scale);

	//后视摄像头创建LUT
	thetax = m_rear_paras.thetax;
	thetay = m_rear_paras.thetay;
	thetaz = m_rear_paras.thetaz;
	xshift = m_rear_paras.xshift+512;
	yshift = m_rear_paras.yshift;
	sf = m_rear_paras.sf;
	scale = m_rear_paras.scale;
	m_rear_mapx = Mat(Size(1280, 1024), CV_32FC1);
	m_rear_mapy = Mat(Size(1280, 1024), CV_32FC1);
	m_rear.Create_UndistortPerspective_LUT(m_rear_mapx,m_rear_mapy,thetax,thetay,thetaz,xshift,yshift,sf,scale);

	//后左视摄像头创建LUT
	thetax = m_rearleft_paras.thetax;
	thetay = m_rearleft_paras.thetay;
	thetaz = m_rearleft_paras.thetaz;
	xshift = m_rearleft_paras.xshift+512;
	yshift = m_rearleft_paras.yshift;
	sf = m_rearleft_paras.sf;
	scale = m_rearleft_paras.scale;
	m_rearleft_mapx = Mat(Size(1280, 1024), CV_32FC1);
	m_rearleft_mapy = Mat(Size(1280, 1024), CV_32FC1);
	m_rearleft.Create_UndistortPerspective_LUT(m_rearleft_mapx,m_rearleft_mapy,thetax,thetay,thetaz,xshift,yshift,sf,scale);

	//后右视摄像头创建LUT
	thetax = m_rearright_paras.thetax;
	thetay = m_rearright_paras.thetay;
	thetaz = m_rearright_paras.thetaz;
	xshift = m_rearright_paras.xshift+512;
	yshift = m_rearright_paras.yshift;
	sf = m_rearright_paras.sf;
	scale = m_rearright_paras.scale;
	m_rearright_mapx = Mat(Size(1280, 1024), CV_32FC1);
	m_rearright_mapy = Mat(Size(1280, 1024), CV_32FC1);
	m_rearright.Create_UndistortPerspective_LUT(m_rearright_mapx,m_rearright_mapy,thetax,thetay,thetaz,xshift,yshift,sf,scale);
}

//void CSurroundView::Cal_SurroundView(Mat& front, Mat& frontleft, Mat& frontright, Mat& rear, Mat& rearleft, Mat& rearright, Mat& dst)
//{
//
//	int height = front.rows;
//	int width = front.cols;
//
//	dst = Mat(Size((int)(width+height*1.5), (int)(width+height)), CV_64FC3);
//
//	size_t memSize_front = front.rows * front.step;				//获取原图大小
//	size_t memSize_frontleft = frontleft.rows * frontleft.step;
//	size_t memSize_frontright = frontright.rows * frontright.step;
//	size_t memSize_rear = rear.rows * rear.step;
//	size_t memSize_rearleft = rearleft.rows * rearleft.step;
//	size_t memSize_rearright = rearright.rows * rearright.step;
//	size_t memSize_dst = dst.rows * dst.step;
//
//	uchar3* cuda_front = NULL;
//	uchar3* cuda_frontleft = NULL;
//	uchar3* cuda_frontright = NULL;
//	uchar3* cuda_rear = NULL;
//	uchar3* cuda_rearleft = NULL;
//	uchar3* cuda_rearright = NULL;
//	uchar3* cuda_dst = NULL;
//	cudaMalloc((void**)&cuda_front, memSize_front);		//分配GPU内存
//	cudaMalloc((void**)&cuda_frontleft, memSize_frontleft);
//	cudaMalloc((void**)&cuda_frontright, memSize_frontright);
//	cudaMalloc((void**)&cuda_rear, memSize_rear);
//	cudaMalloc((void**)&cuda_rearleft, memSize_rearleft);
//	cudaMalloc((void**)&cuda_rearright, memSize_rearright);
//
//	cudaMemcpy(cuda_front, front.data, memSize_front, cudaMemcpyHostToDevice);	//传递数据至GPU显存(原始图像)
//	cudaMemcpy(cuda_frontleft, frontleft.data, memSize_frontleft, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_frontright, frontright.data, memSize_frontright, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rear, rear.data, memSize_rear, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearleft, rearleft.data, memSize_rearleft, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearright, rearright.data, memSize_rearright, cudaMemcpyHostToDevice);
//
//	/*---------------向GPU传递LUT数据，若需要重复使用可考虑优化传递次数，此处未考虑------------------*/
//
//	size_t memSize_frontmap = m_front_mapx.rows * m_front_mapx.step;
//	size_t memSize_frontleftmap = m_frontleft_mapx.rows * m_frontleft_mapx.step;
//	size_t memSize_frontrightmap = m_frontright_mapx.rows * m_frontright_mapx.step;
//	size_t memSize_rearmap = m_rear_mapx.rows * m_rear_mapx.step;
//	size_t memSize_rearleftmap = m_rearleft_mapx.rows * m_rearleft_mapx.step;
//	size_t memSize_rearrightmap = m_rearright_mapx.rows * m_rearright_mapx.step;
//
//	size_t memSize_frontleftcoeff = m_frontleft_coeff.rows * m_frontleft_coeff.step;
//	size_t memSize_frontrightcoeff = m_frontright_coeff.rows * m_frontright_coeff.step;
//	size_t memSize_leftcoeff = m_left_coeff.rows * m_left_coeff.step;
//	size_t memSize_rightcoeff = m_right_coeff.rows * m_right_coeff.step;
//	size_t memSize_rearleftcoeff = m_rearleft_coeff.rows * m_rearleft_coeff.step;
//	size_t memSize_rearrightcoeff = m_rearright_coeff.rows * m_rearright_coeff.step;
//
//	float1* cuda_frontleft_coeff = NULL;
//	float1* cuda_frontright_coeff = NULL;
//	float1* cuda_left_coeff = NULL;
//	float1* cuda_right_coeff = NULL;
//	float1* cuda_rearleft_coeff = NULL;
//	float1* cuda_rearright_coeff = NULL;
//	cudaMalloc((void**)cuda_frontleft_coeff, memSize_frontleftcoeff);
//	cudaMalloc((void**)cuda_frontright_coeff, memSize_frontrightcoeff);
//	cudaMalloc((void**)cuda_left_coeff, memSize_leftcoeff);
//	cudaMalloc((void**)cuda_right_coeff, memSize_rightcoeff);
//	cudaMalloc((void**)cuda_rearleft_coeff, memSize_rearleftcoeff);
//	cudaMalloc((void**)cuda_rearright_coeff, memSize_rearrightcoeff);
//
//	float1* cuda_front_mapx = NULL;
//	float1* cuda_frontleft_mapx = NULL;
//	float1* cuda_frontright_mapx = NULL;
//	float1* cuda_rear_mapx = NULL;
//	float1* cuda_rearleft_mapx = NULL;
//	float1* cuda_rearright_mapx = NULL;
//	float1* cuda_front_mapy = NULL;
//	float1* cuda_frontleft_mapy = NULL;
//	float1* cuda_frontright_mapy = NULL;
//	float1* cuda_rear_mapy = NULL;
//	float1* cuda_rearleft_mapy = NULL;
//	float1* cuda_rearright_mapy = NULL;
//	cudaMalloc((void**)&cuda_front_mapx, memSize_frontmap);		//分配GPU内存
//	cudaMalloc((void**)&cuda_frontleft_mapx, memSize_frontleftmap);
//	cudaMalloc((void**)&cuda_frontright_mapx, memSize_frontright);
//	cudaMalloc((void**)&cuda_rear_mapx, memSize_rearmap);
//	cudaMalloc((void**)&cuda_rearleft_mapx, memSize_rearleftmap);
//	cudaMalloc((void**)&cuda_rearright_mapx, memSize_rearrightmap);
//	cudaMalloc((void**)&cuda_front_mapy, memSize_frontmap);		//分配GPU内存
//	cudaMalloc((void**)&cuda_frontleft_mapy, memSize_frontleftmap);
//	cudaMalloc((void**)&cuda_frontright_mapy, memSize_frontrightmap);
//	cudaMalloc((void**)&cuda_rear_mapy, memSize_rearmap);
//	cudaMalloc((void**)&cuda_rearleft_mapy, memSize_rearleftmap);
//	cudaMalloc((void**)&cuda_rearright_mapy, memSize_rearrightmap);
//
//	cudaMemcpy(cuda_front_mapx, m_front_mapx.data, memSize_frontmap, cudaMemcpyHostToDevice);	//传递数据至GPU显存(查找表)
//	cudaMemcpy(cuda_frontleft_mapx, m_frontleft_mapx.data, memSize_frontleftmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_frontright_mapx, m_frontright_mapx.data, memSize_frontrightmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rear_mapx, m_rear_mapx.data, memSize_rearmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearleft_mapx, m_rearleft_mapx.data, memSize_rearleftmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearright_mapx, m_rearright_mapx.data, memSize_rearrightmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_front_mapy, m_front_mapy.data, memSize_frontmap, cudaMemcpyHostToDevice);	//传递数据至GPU显存
//	cudaMemcpy(cuda_frontleft_mapy, m_frontleft_mapy.data, memSize_frontleftmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_frontright_mapy, m_frontright_mapy.data, memSize_frontrightmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rear_mapy, m_rear_mapy.data, memSize_rearmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearleft_mapy, m_rearleft_mapy.data, memSize_rearleftmap, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearright_mapy, m_rearright_mapy.data, memSize_rearrightmap, cudaMemcpyHostToDevice);
//
//	cudaMemcpy(cuda_frontleft_coeff, m_frontleft_coeff.data, memSize_frontleftcoeff, cudaMemcpyHostToDevice);	//传递数据至缓存(图像拼接系数)
//	cudaMemcpy(cuda_frontright_coeff, m_frontright_coeff.data, memSize_frontrightcoeff, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_left_coeff, m_left_coeff.data, memSize_leftcoeff, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_right_coeff, m_right_coeff.data, memSize_rightcoeff, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearleft_coeff, m_rearleft_coeff.data, memSize_rearleftcoeff, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_rearright_coeff, m_rearright_coeff.data, memSize_rearrightcoeff, cudaMemcpyHostToDevice);
//
//	/*-----------------------------------------------------------------------------------------------*/
//
//	cuda_SurroundView_Caller(cuda_front,cuda_front_mapx,cuda_front_mapx,		//执行回调函数，调用GPU
//								cuda_frontleft,cuda_frontleft_mapx,cuda_frontleft_mapx, 
//								cuda_frontright,cuda_frontright_mapx,cuda_frontright_mapx, 
//								cuda_rear,cuda_rear_mapx,cuda_rear_mapx, 
//								cuda_rearleft,cuda_rearleft_mapx,cuda_rearleft_mapx, 
//								cuda_rearright,cuda_rearright_mapx,cuda_rearright_mapx, 
//								cuda_frontleft_coeff,cuda_frontright_coeff,cuda_left_coeff,cuda_right_coeff,cuda_rearleft_coeff,cuda_rearright_coeff,
//								height, width,
//								cuda_dst);
//
//	cudaMemcpy(dst.data, cuda_dst, memSize_dst, cudaMemcpyDeviceToHost);	//得到环视图像
//
//	cudaFree(cuda_front);		//释放GPU显存
//	cudaFree(cuda_frontleft);
//	cudaFree(cuda_frontright);
//	cudaFree(cuda_rear);
//	cudaFree(cuda_rearleft);
//	cudaFree(cuda_rearright);
//
//	/*---------------向GPU传递LUT数据，若需要重复使用可考虑优化传递次数，此处未考虑------------------*/
//	cudaFree(cuda_front_mapx);
//	cudaFree(cuda_frontleft_mapx);
//	cudaFree(cuda_frontright_mapx);
//	cudaFree(cuda_rear_mapx);
//	cudaFree(cuda_rearleft_mapx);
//	cudaFree(cuda_rearright_mapx);
//
//	cudaFree(cuda_front_mapy);
//	cudaFree(cuda_frontleft_mapy);
//	cudaFree(cuda_frontright_mapy);
//	cudaFree(cuda_rear_mapy);
//	cudaFree(cuda_rearleft_mapy);
//	cudaFree(cuda_rearright_mapy);
//
//	cudaFree(cuda_frontleft_coeff);
//	cudaFree(cuda_frontright_coeff);
//	cudaFree(cuda_left_coeff);
//	cudaFree(cuda_right_coeff);
//	cudaFree(cuda_rearleft_coeff);
//	cudaFree(cuda_rearright_coeff);
//
//	/*-----------------------------------------------------------------------------------------------*/
//}

void imageMulcoeff(Mat& image, Mat& coeff, Mat& dstImage)
{
	int height = image.rows;
	int width = image.cols;

	size_t memSize_image = height * image.step;
	size_t memSize_coeff = coeff.rows * coeff.step;

	float3* cuda_image = NULL;
	float3* cuda_dstImage = NULL;
	float1* cuda_coeff = NULL;

	cudaMalloc((void**)&cuda_image, memSize_image);
	cudaMalloc((void**)&cuda_dstImage, memSize_image);
	cudaMalloc((void**)&cuda_coeff, memSize_coeff);

	cudaMemcpy(cuda_image, image.data, memSize_image, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_coeff, coeff.data, memSize_coeff, cudaMemcpyHostToDevice);

	cuda_imageMulcoeff_Caller(cuda_image, cuda_coeff, cuda_dstImage, width, height);

	cudaMemcpy(dstImage.data, cuda_dstImage,memSize_image, cudaMemcpyDeviceToHost);

	cudaFree(cuda_image);
	cudaFree(cuda_coeff);
	cudaFree(cuda_dstImage);
}

//void CSurroundView::Cal_SurroundView(Mat& front, Mat& frontleft, Mat& frontright, Mat& rear, Mat& rearleft, Mat& rearright, Mat& dst)
//{
//	//imshow("front", front);
//	//waitKey();
//	//Mat frontROI(front, Rect(100,100,500,500));
//	//imshow("frontROI", frontROI);
//	//waitKey();
//
//	Mat dst_front = front;			//计算视角变换后的图像
//	remap(front, dst_front, m_front_mapx, m_front_mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);	/*2015/02/10问题：执行完remap函数后frontROI改变？*/
//	//imshow("frontROI2", frontROI);
//	//waitKey();
//	//imshow("dst_front", dst_front);
//	//imwrite("dst_front.bmp", dst_front);
//	//waitKey();
//
//	Mat front_mosaic;
//	imageMulcoeff(dst_front,m_front_coeff,front_mosaic);
//	imshow("front_mosaic",front_mosaic);
//	waitKey();
//
//	Mat dst_frontleft = frontleft;
//	remap(frontleft, dst_frontleft, m_frontleft_mapx, m_frontleft_mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
//
//	Mat dst_frontright = frontright;
//	remap(frontright, dst_frontright, m_frontright_mapx, m_frontright_mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
//
//	Mat dst_rear = rear;
//	remap(rear, dst_rear, m_rear_mapx, m_rear_mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
//
//	Mat dst_rearleft = rearleft;
//	remap(rearleft, dst_rearleft, m_rearleft_mapx, m_rearleft_mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
//
//	Mat dst_rearright = rearright;
//	remap(rearright, dst_rearright, m_rearright_mapx, m_rearright_mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
//
//	Mat dstROI(dst,Rect(10,10,dst_front.cols, dst_front.rows));
//	//dst_front.copyTo(dstROI);
//	dstROI = dstROI+dst_front;
//	//Mat dstROI2(dst,Rect(130,130,dst_front.cols, dst_front.rows));
//	//dst_front.copyTo(dstROI2);
//	//Mat dst_frontROI(dst_front, Rect(10,10,500,500));
//	//imshow("frontROI3", frontROI);
//	//waitKey();
//	//frontROI = front(Rect(50,50,100,100));
//	//frontROI.convertTo(dst_front(Rect(50,50,100,100)), dst_front.type());
//	//dst_frontROI = dst_front;
//	//dst_front(Rect(0,0,100,100)) = front(Rect(0,0,100,100));
//	//dst_frontROI = frontROI;
//	//frontROI.copyTo(dst_frontROI);
//	//imshow("frontROI4", frontROI);
//	//waitKey();
//	imshow("surround",dst);
//	imwrite("surround.bmp", dst);
//	waitKey();
//
//	//Mat front_coeff3 = Mat(Size(m_front_coeff.cols, m_front_coeff.rows), CV_32FC3);
//	//cvtColor(m_front_coeff, front_coeff3, CV_GRAY2RGB);
//	//Mat front_mosaic = front;
//	//multiply(dst_front, m_front_coeff, front_mosaic);
//}

void CSurroundView::Get_LUT(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright)
{
	m_front_LUT = imread(front, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	//ushort3* ptr = m_front_LUT.ptr<ushort3>(0);
	//uchar3* ptr = m_front_LUT.ptr<uchar3>(0);
	m_frontleft_LUT = imread(frontleft,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	m_frontright_LUT = imread(frontright,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	m_rear_LUT = imread(rear,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	m_rearleft_LUT = imread(rearleft,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	m_rearright_LUT = imread(rearright,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
}

void Get_remapPic(Mat& src, Mat& dst, Mat& map)
{
	int width = map.cols;
	int height = map.rows;
	int channels = map.channels();

	int nRows = height;
	int nCols = width;

	int nWidth = src.cols;		//原图的宽度
	int nHeight = src.rows;

	dst = Mat(Size(width, height), src.type());


	if(map.isContinuous() && dst.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	ushort3* ptr1;
	uchar3* ptr2;
	uchar3* ptr3 = src.ptr<uchar3>(0);
	for (int i = 0; i < nRows; i++)
	{
		ptr1 = map.ptr<ushort3>(i);
		ptr2 = dst.ptr<uchar3>(i);
		for (int j = 0; j < nCols; j++)
		{
			//ptr2[j].x = src[j].x;
			//ptr2[j].y = src[j].y;
			//ptr2[j].z = src[j].z;
			if (ptr1[j].z>=1 && ptr1[j].z<=nHeight && ptr1[j].y>=1 && ptr1[j].y<=nWidth)
			{
				ptr2[j].x = ptr3[(ptr1[j].z-1)*nWidth+ptr1[j].y-1].x*(ptr1[j].x/1000.0);
				ptr2[j].y = ptr3[(ptr1[j].z-1)*nWidth+ptr1[j].y-1].y*(ptr1[j].x/1000.0);
				ptr2[j].z = ptr3[(ptr1[j].z-1)*nWidth+ptr1[j].y-1].z*(ptr1[j].x/1000.0);
			}
			else
			{
				ptr2[j].x = 0;
				ptr2[j].y = 0;
				ptr2[j].z = 0;
			}
		}
	}
}

void CSurroundView::Cal_SurroundView(Mat& front, Mat& frontleft, Mat& frontright, Mat& rear, Mat& rearleft, Mat& rearright, Mat& dst)
{
	int width = front.cols;
	int height = front.rows;

	int d_width = dst.cols;
	int d_height = dst.rows;

	Mat front_dst, frontleft_dst, frontright_dst, rear_dst, rearleft_dst, rearright_dst;

	Get_remapPic(front, front_dst, m_front_LUT);			//得到矫正后的待拼接图像
	Get_remapPic(frontleft, frontleft_dst, m_frontleft_LUT);
	Get_remapPic(frontright, frontright_dst, m_frontright_LUT);
	Get_remapPic(rear, rear_dst, m_rear_LUT);
	Get_remapPic(rearleft, rearleft_dst, m_rearleft_LUT);
	Get_remapPic(rearright, rearright_dst, m_rearright_LUT);

	Mat dstROI = Mat(dst, Rect(height/2,0,width,height));
	dstROI = dstROI + front_dst;

	dstROI = Mat(dst, Rect(0,height/2,height,width));
	dstROI = dstROI + frontleft_dst;

	dstROI = Mat(dst, Rect(d_width-height,height/2,height,width));
	dstROI = dstROI + frontright_dst;

	dstROI = Mat(dst, Rect(0,(height+width)/2,height,width));
	dstROI = dstROI + rearleft_dst;

	dstROI = Mat(dst, Rect(d_width-height,(height+width)/2,height,width));
	dstROI = dstROI + rearright_dst;

	dstROI = Mat(dst, Rect(height/2,width+height/2,width,height));			//2015/02/11未解决：此处由于Matlab的疏忽，导致拼接区域有些变动
	dstROI = dstROI + rear_dst;
	//int width = m_front_LUT.cols;
	//int height = m_front_LUT.rows;
	//int channels = m_front_LUT.channels();

	//int nWidth = 1280;		//原图的宽度
	//int nHeight = 1024;

	//int nRows = height;
	//int nCols = width;

	//Mat front_dst = Mat(Size(width, height), front.type());


	//if(m_front_LUT.isContinuous() && front_dst.isContinuous())
	//{
	//	nCols *= nRows;
	//	nRows = 1;
	//}

	//ushort3* ptr1;
	//uchar3* ptr2;
	//uchar3* src = front.ptr<uchar3>(0);
	//for (int i = 0; i < nRows; i++)
	//{
	//	ptr1 = m_front_LUT.ptr<ushort3>(i);
	//	ptr2 = front_dst.ptr<uchar3>(i);
	//	for (int j = 0; j < nCols; j++)
	//	{
	//		//ptr2[j].x = src[j].x;
	//		//ptr2[j].y = src[j].y;
	//		//ptr2[j].z = src[j].z;
	//		if (ptr1[j].z>=1 && ptr1[j].z<=nHeight && ptr1[j].y>=1 && ptr1[j].y<=nWidth)
	//		{
	//			ptr2[j].x = src[(ptr1[j].z-1)*nWidth+ptr1[j].y-1].x*(ptr1[j].x/1000.0);
	//			ptr2[j].y = src[(ptr1[j].z-1)*nWidth+ptr1[j].y-1].y*(ptr1[j].x/1000.0);
	//			ptr2[j].z = src[(ptr1[j].z-1)*nWidth+ptr1[j].y-1].z*(ptr1[j].x/1000.0);
	//		}
	//	}
	//}

	//imwrite("front_test.bmp",front_dst);
	//imshow("front_test.bmp",front_dst);
}
