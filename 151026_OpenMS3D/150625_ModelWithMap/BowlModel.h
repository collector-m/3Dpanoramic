

#ifndef _BOWLMODEL_H_
#define _BOWLMODEL_H_

#include <GL/glut.h>

#include "opencv2/opencv.hpp"

//	纹理ID
GLuint texttureID[4];

//	贴图图片
cv::Mat Front, Left, Right, Rear;
cv::Mat FrontMap, LeftMap, RightMap, RearMap;

enum drawtype	//模型绘制类型（SOLID:方格；WIRE:线型）
{
	SOLID,
	WIRE
};

void initTexture(const char* filename1, const char* filename2, const char* filename3, const char* filename4);

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type);

#endif