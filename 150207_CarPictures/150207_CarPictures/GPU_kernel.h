
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//�ӽǱ任��GPU���ó���
void cuda_Perspective_Caller(const uchar3 *src, uchar3 *dst, int height, int width, int height_dst, int width_dst, float* H, float scale);

//���۾�ͷ���������GPU���ó���
void cuda_Undistort_Caller(float1* mapx, float1* mapy, struct ocam_model *m_model, int d_width, int d_height, 
								float Nxc, float Nyc, float Nz, float xshift, float yshift, float sf, float scale);

//���۾�ͷ�ӽǱ任��GPU���ó���
void cuda_UndistortPerspective_Caller(float1* mapx, float1* mapy, struct ocam_model *m_model, int d_width, int d_height, 
											float Nxc, float Nyc, float Nz, float* H, float xshift, float yshift, float sf, float scale);

////���㻷��ͼ��ص�����
//void cuda_SurroundView_Caller(const uchar3 *front,const float1 *front_mapx,const float1 *front_mapy,
//								const uchar3 *frontleft,const float1 *frontleft_mapx,const float1 *frontleft_mapy,
//								const uchar3 *frontright,const float1 *frontright_mapx,const float1 *frontright_mapy,
//								const uchar3 *rear,const float1 *rear_mapx,const float1 *rear_mapy,
//								const uchar3 *rearleft,const float1 *rearleft_mapx,const float1 *rearleft_mapy,
//								const uchar3 *rearright,const float1 *rearright_mapx,const float1 *rearright_mapy,
//								const float1 *frontleft_coeff,const float1 *frontright_coeff,const float1 *left_coeff,const float1 *right_coeff,const float1 *rearleft_coeff,const float1 *rearright_coeff,
//								int height, int width,
//								uchar3 *dst);

//����ͼ����ϵ����˵��ú���
void cuda_imageMulcoeff_Caller(const float3 *src, const float1 *coeff, float3 *dst, int width, int height);