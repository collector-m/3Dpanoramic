

#ifndef _BOWLMODEL_H_
#define _BOWLMODEL_H_

#include <GL/glut.h>

enum drawtype	//ģ�ͻ������ͣ�SOLID:����WIRE:���ͣ�
{
	SOLID,
	WIRE
};

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type);

#endif