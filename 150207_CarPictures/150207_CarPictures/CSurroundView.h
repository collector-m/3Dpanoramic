

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
	CFisheyeLens m_front;		//������������ͷ
	CFisheyeLens m_frontleft;
	CFisheyeLens m_frontright;
	CFisheyeLens m_rear;
	CFisheyeLens m_rearleft;
	CFisheyeLens m_rearright;

	Mat m_frontleft_coeff;		//�ں������ϵ������
	Mat m_frontright_coeff;
	Mat m_rearleft_coeff;
	Mat m_rearright_coeff;
	Mat m_front_coeff;
	Mat m_rear_coeff;

	struct Per_Paras m_front_paras;		//������������ͷ���ӽǱ任����
	struct Per_Paras m_frontleft_paras;
	struct Per_Paras m_frontright_paras;
	struct Per_Paras m_rear_paras;
	struct Per_Paras m_rearleft_paras;
	struct Per_Paras m_rearright_paras;

	Mat m_front_mapx, m_front_mapy;		//������������ͷ���ӽǱ任���ұ�
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
	/*---------------------2015/02/10��һ�棬δʹ�ò��ұ�------------------------------------*/

	//��ʼ����������ȡ��������ģ�͵Ĳ���
	bool Get_ocammodel(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright);
	//��ȡͼ����ں�ϵ��
	void Get_blendcoeff(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright);
	//��ȡ�ӽǱ任����
	void Get_persparas(char *filename);
	//�����������۵��ӽǱ任���ұ�
	void Create_6_UndistortPerspective_LUT();
	//���㻷��ͼ��
	void Cal_SurroundView(Mat& front, Mat& frontleft, Mat& frontright, Mat& rear, Mat& rearleft, Mat& rearright, Mat& dst);

	/*----------------------------------------------------------------------------------------*/

	/*------------------2015/02/10�ڶ��棺ʹ�ò��ұ�------------------------------------------*/

	//��ʼ����������ȡ���ұ�
	void Get_LUT(char *front, char *frontleft, char *frontright, char *rear, char *rearleft, char *rearright);
};