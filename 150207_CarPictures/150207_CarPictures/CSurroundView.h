

#include "CFisheyeLens.h"

struct Per_Paras
{
	float thetax;
	float thetay;
	float thetaz;

	float xshift;
	float yshift;
	float sf;
	float scale;
};

class CSurroundView
{
public:
	CSurroundView();
	~CSurroundView();

public:
	CFisheyeLens m_front;		//六个鱼眼摄像头
	CFisheyeLens m_frontleft;
	CFisheyeLens m_frontright;
	CFisheyeLens m_rear;
	CFisheyeLens m_rearleft;
	CFisheyeLens m_rearright;

	Mat m_frontleft_coeff;		//融合区域的系数矩阵
	Mat m_frontright_coeff;
	Mat m_rearleft_coeff;
	Mat m_rearright_coeff;
	Mat m_front_coeff;
	Mat m_rear_coeff;

	struct Per_Paras m_front_paras;		//六个鱼眼摄像头的视角变换参数
	struct Per_Paras m_frontleft_paras;
	struct Per_Paras m_frontright_paras;
	struct Per_Paras m_rear_paras;
	struct Per_Paras m_rearleft_paras;
	struct Per_Paras m_rearright_paras;

	Mat m_front_mapx, m_front_mapy;		//六个鱼眼摄像头的视角变换查找表
	Mat m_frontleft_mapx, m_frontleft_mapy;
	Mat m_frontright_mapx, m_frontright_mapy;
	Mat m_rear_mapx, m_rear_mapy;
	Mat m_rearleft_mapx, m_rearleft_mapy;
	Mat m_rearright_mapx, m_rearright_mapy;

	Mat m_front_LUT;
	Mat m_frontleft_LUT;
	Mat m_frontright_LUT;
	Mat m_rear_LUT;
	Mat m_rearleft_LUT;
	Mat m_rearright_LUT;

public:
	/*---------------------2015/02/10第一版，未使用查找表------------------------------------*/

	//初始化函数，读取六个鱼眼模型的参数
	bool Get_ocammodel(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright);
	//读取图像的融合系数
	void Get_blendcoeff(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright);
	//读取视角变换参数
	void Get_persparas(char *filename);
	//创建六个鱼眼的视角变换查找表
	void Create_6_UndistortPerspective_LUT();
	//计算环视图像
	void Cal_SurroundView(Mat& front, Mat& frontleft, Mat& frontright, Mat& rear, Mat& rearleft, Mat& rearright, Mat& dst);

	/*----------------------------------------------------------------------------------------*/

	/*------------------2015/02/10第二版：使用查找表------------------------------------------*/

	//初始化函数，读取查找表
	void Get_LUT(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright);
};