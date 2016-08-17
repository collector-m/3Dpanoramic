

#include "CFisheyeLens.h"
#include "GPU_kernel.h"

#define PI 3.14159


CFisheyeLens::CFisheyeLens()
{

}

CFisheyeLens::~CFisheyeLens()
{

}

/*------------------------------------------------------------------------------
 This function reads the parameters of the omnidirectional camera model from 
 a given TXT file
------------------------------------------------------------------------------*/
//------------------------------------------------------------------------------
int get_ocam_model(struct ocam_model *myocam_model, char *filename)
{
	double *pol        = myocam_model->pol;
	double *invpol     = myocam_model->invpol; 
	double *xc         = &(myocam_model->xc);
	double *yc         = &(myocam_model->yc); 
	double *c          = &(myocam_model->c);
	double *d          = &(myocam_model->d);
	double *e          = &(myocam_model->e);
	int    *width      = &(myocam_model->width);
	int    *height     = &(myocam_model->height);
	int *length_pol    = &(myocam_model->length_pol);
	int *length_invpol = &(myocam_model->length_invpol);
	FILE *f;
	char buf[CMV_MAX_BUF];
	int i;

	//Open file
	if(!(f=fopen(filename,"r")))
	{
		printf("File %s cannot be opened\n", filename);				  
		return -1;
	}

	//Read polynomial coefficients
	fgets(buf,CMV_MAX_BUF,f);
	fscanf(f,"\n");
	fscanf(f,"%d", length_pol);
	for (i = 0; i < *length_pol; i++)
	{
		fscanf(f," %lf",&pol[i]);
	}

	//Read inverse polynomial coefficients
	fscanf(f,"\n");
	fgets(buf,CMV_MAX_BUF,f);
	fscanf(f,"\n");
	fscanf(f,"%d", length_invpol);
	for (i = 0; i < *length_invpol; i++)
	{
		fscanf(f," %lf",&invpol[i]);
	}

	//Read center coordinates
	fscanf(f,"\n");
	fgets(buf,CMV_MAX_BUF,f);
	fscanf(f,"\n");
	fscanf(f,"%lf %lf\n", xc, yc);

	//Read affine coefficients
	fgets(buf,CMV_MAX_BUF,f);
	fscanf(f,"\n");
	fscanf(f,"%lf %lf %lf\n", c,d,e);

	//Read image size
	fgets(buf,CMV_MAX_BUF,f);
	fscanf(f,"\n");
	fscanf(f,"%d %d", height, width);

	fclose(f);
	return 0;
}

int CFisheyeLens::Initial(char *filename)
{
	return get_ocam_model(&m_model, filename);
}

/*------------------------------------------------------------------------------
 CAM2WORLD projects a 2D point onto the unit sphere
    CAM2WORLD(POINT3D, POINT2D, OCAM_MODEL) 
    back-projects a 2D point (point2D), in pixels coordinates, 
    onto the unit sphere returns the normalized coordinates point3D = [x;y;z]
    where (x^2 + y^2 + z^2) = 1.
    
    POINT3D = [X;Y;Z] are the coordinates of the 3D points, such that (x^2 + y^2 + z^2) = 1.
    OCAM_MODEL is the model of the calibrated camera.
    POINT2D = [rows;cols] are the pixel coordinates of the point in pixels
    
    Copyright (C) 2009 DAVIDE SCARAMUZZA   
    Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org
  
    NOTE: the coordinates of "point2D" and "center" are already according to the C
    convention, that is, start from 0 instead than from 1.
------------------------------------------------------------------------------*/
//------------------------------------------------------------------------------
void world2cam(double point2D[2], double point3D[3], struct ocam_model *myocam_model)
{
 double *invpol     = myocam_model->invpol; 
 double xc          = (myocam_model->xc);
 double yc          = (myocam_model->yc); 
 double c           = (myocam_model->c);
 double d           = (myocam_model->d);
 double e           = (myocam_model->e);
 int    width       = (myocam_model->width);
 int    height      = (myocam_model->height);
 int length_invpol  = (myocam_model->length_invpol);
 double norm        = sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1]);
 double theta       = atan(point3D[2]/norm); //*sf
 double t, t_i;
 double rho, x, y;
 double invnorm;
 int i;
  
  if (norm != 0) 
  {
    invnorm = 1/norm;
    t  = theta;
    rho = invpol[0];
    t_i = 1;

    for (i = 1; i < length_invpol; i++)
    {
      t_i *= t;
      rho += t_i*invpol[i];
    }

    x = point3D[0]*invnorm*rho;
    y = point3D[1]*invnorm*rho;
  
    point2D[0] = x*c + y*d + xc;
    point2D[1] = x*e + y   + yc;
  }
  else
  {
    point2D[0] = xc;
    point2D[1] = yc;
  }
}

Point CFisheyeLens::UndistorPoint(Point src, float sf, float scale)
{
	return src;
}

Point CFisheyeLens::UndistorPrespectivePoint(Point src, float thetax, float thetay, float thetaz, float xshift, float yshift, float sf, float scale)
{
	return src;
}

void CFisheyeLens::Create_Undistort_LUT(Mat& mapx, Mat& mapy, float xshift, float yshift, float sf, float scale)
{
	int width = mapx.cols;		//获取目标图像的长宽
	int height = mapx.rows;

	float Nxc = height/2.0;
	float Nyc = width/2.0;
	float Nz = -width/sf;

	size_t memSize = height * mapx.step;

	float1* cuda_mapx = NULL;
	float1* cuda_mapy = NULL;
	struct ocam_model *cuda_model = NULL;
	cudaMalloc((void**)&cuda_mapx, memSize);
	cudaMalloc((void**)&cuda_mapy, memSize);
	cudaMalloc((void**)&cuda_model, sizeof(ocam_model));

	cudaMemcpy(cuda_model, &(m_model), sizeof(ocam_model), cudaMemcpyHostToDevice);

	cuda_Undistort_Caller(cuda_mapx, cuda_mapy, cuda_model, width, height, Nxc, Nyc, Nz, xshift, yshift, sf, scale);

	cudaMemcpy(mapx.data, cuda_mapx, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mapy.data, cuda_mapy, memSize, cudaMemcpyDeviceToHost);

	cudaFree(cuda_mapx);
	cudaFree(cuda_mapy);
	cudaFree(cuda_model);
}

void CFisheyeLens::Create_UndistortPerspective_LUT(Mat& mapx, Mat& mapy, float thetax, float thetay, float thetaz, float xshift, float yshift, float sf, float scale)
{
	int width = mapx.cols;		//获取目标图像的长宽
	int height = mapx.rows;

	float Nxc = height/2.0;
	float Nyc = width/2.0;
	//float Nz = -width/sf;

	thetax = thetax/180*PI;	//角度转为弧度
	thetay = thetay/180*PI;
	thetaz = thetaz/180*PI;

	float Nz = cos(thetax)*cos(thetay)*cos(thetaz);

	float f = 400;
	//Mat T = (Mat_<float>(3,3)<<	//平移矩阵
	//	1,		0,	xshift/sf,
	//	0,		1,	yshift/sf,
	//	0,		0,		1);
	Mat A1 = (Mat_<float>(3,3)<<	
		f,		0,		0,
		0,		f,		0,
		0,		0,		1);
	Mat A2 = (Mat_<float>(3,3)<<	
		1/f,	0,		0,
		0,		1/f,	0,
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
	//Mat H = A1*R*T*A2;		//变换矩阵
	Mat H = A1*R*A2;		//变换矩阵

	float Ht[9] = {0};
	for (int i = 0; i<3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Ht[i*3+j] = H.at<float>(i,j);
		}
	}

	size_t memSize = height * mapx.step;

	float1* cuda_mapx = NULL;
	float1* cuda_mapy = NULL;
	float* cuda_H = NULL;
	struct ocam_model *cuda_model = NULL;
	cudaMalloc((void**)&cuda_mapx, memSize);
	cudaMalloc((void**)&cuda_mapy, memSize);
	cudaMalloc((void**)&cuda_H, 9*sizeof(float));
	cudaMalloc((void**)&cuda_model, sizeof(ocam_model));

	cudaMemcpy(cuda_model, &(m_model), sizeof(ocam_model), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_H, Ht, 9*sizeof(float), cudaMemcpyHostToDevice);

	cuda_UndistortPerspective_Caller(cuda_mapx, cuda_mapy, cuda_model, width, height, Nxc, Nyc, Nz, cuda_H, xshift, yshift ,sf, scale);

	cudaMemcpy(mapx.data, cuda_mapx, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mapy.data, cuda_mapy, memSize, cudaMemcpyDeviceToHost);

	cudaFree(cuda_mapx);
	cudaFree(cuda_mapy);
	cudaFree(cuda_H);
	cudaFree(cuda_model);
}