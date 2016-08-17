

#ifndef _BOWLMODEL_H_
#define _BOWLMODEL_H_

#include <GL/glut.h>

enum drawtype	//模型绘制类型（SOLID:方格；WIRE:线型）
{
	SOLID,
	WIRE
};

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type);

#endif