/*
*/

#include "opencv.h"

class CNormalLens
{
public:
	CNormalLens();
	~CNormalLens();

public:
	//����ӽǱ任
	Point PerspectivePoint(Point src, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale);
	//ͼ����ӽǱ任
	void PerspectiveImage(Mat &src, Mat &dst, float thetax, float thetay, float thetaz, float xshift, float yshift, float scale);
	//�����ӽǱ任���ұ�
	void Create_Perspective_LUT(Mat &mapx, Mat &mapy, double thetax, double thetay, double thetaz, double xshift, double yshift, float scale);
};