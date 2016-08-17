/*------------------------------------------------------------------------------
Copyright (C) 2008 DAVIDE SCARAMUZZA, ETH Zurich
Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
	2015/06/25: ocam_functions.h	by:��ʫ��

	���ܣ���ȡ���ģ�ͣ��������������Ϊ��������
------------------------------------------------------------------------------*/

const int CMV_MAX_BUF = 1024;
const int MAX_POL_LENGTH = 64;

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


/*------------------------------------------------------------------------------
This function reads the parameters of the omnidirectional camera model from
a given TXT file
------------------------------------------------------------------------------*/
int get_ocam_model(struct ocam_model *myocam_model, char *filename);

/*------------------------------------------------------------------------------
WORLD2CAM projects a 3D point on to the image
WORLD2CAM(POINT2D, POINT3D, OCAM_MODEL)
projects a 3D point (point3D) on to the image and returns the pixel coordinates (point2D).

POINT3D = [X;Y;Z] are the coordinates of the 3D point.
OCAM_MODEL is the model of the calibrated camera.
POINT2D = [rows;cols] are the pixel coordinates of the reprojected point

Copyright (C) 2009 DAVIDE SCARAMUZZA
Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org

NOTE: the coordinates of "point2D" and "center" are already according to the C
convention, that is, start from 0 instead than from 1.
------------------------------------------------------------------------------*/
void world2cam(double point2D[2], double point3D[3], struct ocam_model *myocam_model);

