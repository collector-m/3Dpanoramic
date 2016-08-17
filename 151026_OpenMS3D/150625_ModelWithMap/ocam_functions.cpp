/*------------------------------------------------------------------------------
Copyright (C) 2008 DAVIDE SCARAMUZZA, ETH Zurich
Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
2015/06/25: ocam_functions.cpp	by:胡诗卉

功能：读取相机模型，将世界坐标成像为像素坐标
------------------------------------------------------------------------------*/

#include "ocam_functions.h"

#include <stdlib.h>
#include <stdio.h>

#include <math.h>

//------------------------------------------------------------------------------
int get_ocam_model(struct ocam_model *myocam_model, char *filename)
{
	//double *pol = myocam_model->pol;
	//double *invpol = myocam_model->invpol;
	//double *xc = &(myocam_model->xc);
	//double *yc = &(myocam_model->yc);
	//double *c = &(myocam_model->c);
	//double *d = &(myocam_model->d);
	//double *e = &(myocam_model->e);
	//int    *width = &(myocam_model->width);
	//int    *height = &(myocam_model->height);
	//int *length_pol = &(myocam_model->length_pol);
	//int *length_invpol = &(myocam_model->length_invpol);
	//FILE *f;
	//char buf[CMV_MAX_BUF];
	//int i;

	////Open file
	//if (!(f = fopen(filename, "r")))
	//{
	//	printf("File %s cannot be opened\n", filename);
	//	return -1;
	//}

	////Read polynomial coefficients
	//fgets(buf, CMV_MAX_BUF, f);
	//fscanf(f, "\n");
	//fscanf(f, "%d", length_pol);
	//for (i = 0; i < *length_pol; i++)
	//{
	//	fscanf(f, " %lf", &pol[i]);
	//}

	////Read inverse polynomial coefficients
	//fscanf(f, "\n");
	//fgets(buf, CMV_MAX_BUF, f);
	//fscanf(f, "\n");
	//fscanf(f, "%d", length_invpol);
	//for (i = 0; i < *length_invpol; i++)
	//{
	//	fscanf(f, " %lf", &invpol[i]);
	//}

	////Read center coordinates
	//fscanf(f, "\n");
	//fgets(buf, CMV_MAX_BUF, f);
	//fscanf(f, "\n");
	//fscanf(f, "%lf %lf\n", xc, yc);

	////Read affine coefficients
	//fgets(buf, CMV_MAX_BUF, f);
	//fscanf(f, "\n");
	//fscanf(f, "%lf %lf %lf\n", c, d, e);

	////Read image size
	//fgets(buf, CMV_MAX_BUF, f);
	//fscanf(f, "\n");
	//fscanf(f, "%d %d", height, width);

	//fclose(f);
	return 0;
}

//------------------------------------------------------------------------------
void world2cam(double point2D[2], double point3D[3], struct ocam_model *myocam_model)
{
	double *invpol = myocam_model->invpol;
	double xc = (myocam_model->xc);
	double yc = (myocam_model->yc);
	double c = (myocam_model->c);
	double d = (myocam_model->d);
	double e = (myocam_model->e);
	int    width = (myocam_model->width);
	int    height = (myocam_model->height);
	int length_invpol = (myocam_model->length_invpol);
	double norm = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
	double theta = atan(point3D[2] / norm);
	double t, t_i;
	double rho, x, y;
	double invnorm;
	int i;

	if (norm != 0)
	{
		invnorm = 1 / norm;
		t = theta;
		rho = invpol[0];
		t_i = 1;

		for (i = 1; i < length_invpol; i++)
		{
			t_i *= t;
			rho += t_i*invpol[i];
		}

		x = point3D[0] * invnorm*rho;
		y = point3D[1] * invnorm*rho;

		point2D[0] = x*c + y*d + xc;
		point2D[1] = x*e + y + yc;
	}
	else
	{
		point2D[0] = xc;
		point2D[1] = yc;
	}
}