/*
*/

#include "opencv.h"

class CNormalLens
{
public:
	CNormalLens();
	~CNormalLens();

public:
	//点的视角变换
	Point PerspectivePoint(Point src, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale);
	//图像的视角变换
	void PerspectiveImage(Mat &src, Mat &dst, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale);
	//创建视角变换查找表
	void Create_Perspective_LUT(Mat &mapx, Mat &mapy, double thetax, double thetay, double thetaz, double xshift, double yshift, float scale);
};