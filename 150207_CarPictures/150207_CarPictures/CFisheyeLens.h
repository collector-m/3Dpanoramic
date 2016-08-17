

#include "opencv.h"

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

class CFisheyeLens
{

public:
	CFisheyeLens();
	~CFisheyeLens();

public:
	int Initial(char *filename);

public:
	ocam_model m_model;

public:
	Point UndistorPoint(Point src, float sf, float scale);	//畸变矫正点

	Point UndistorPrespectivePoint(Point src, float thetax, float thetay, float thetaz, float xshift, float yshift, float sf, float scale);	//视角变换点

	void Create_Undistort_LUT(Mat& mapx, Mat& mapy, float xshift, float yshift, float sf, float scale);//创建畸变矫正查找表

	void Create_UndistortPerspective_LUT(Mat& mapx, Mat& mapy, float thetax, float thetay, float thetaz, float xshift, float yshift, float sf, float scale);//创建视角变换查找表
};
