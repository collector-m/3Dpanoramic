

#ifndef _BOWLMODEL_H_
#define _BOWLMODEL_H_

#include <GL/glut.h>

#include "opencv2/opencv.hpp"

//	����ID
GLuint texttureID[4];

//	��ͼͼƬ
cv::Mat Front, Left, Right, Rear;
cv::Mat FrontMap, LeftMap, RightMap, RearMap;

enum drawtype	//ģ�ͻ������ͣ�SOLID:����WIRE:���ͣ�
{
	SOLID,
	WIRE
};

void initTexture(const char* filename1, const char* filename2, const char* filename3, const char* filename4);

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type);

#endif