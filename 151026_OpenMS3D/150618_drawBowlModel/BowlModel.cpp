

#include "BowlModel.h"

#include <vector>

const double pi = 3.1415926;

typedef struct Point3f		//球体模型顶点
{
	GLfloat x;
	GLfloat y;
	GLfloat z;
}point;

void getPointBottom(point& temp, float a, float b)
{
	temp.x = a*cos(b*pi / 180.0);
	temp.y = a*sin(b*pi / 180.0);
	temp.z = 0;
}

void getPointSide(point& temp, GLfloat length, float a, float b)
{
	temp.x = length*sin(a*pi / 180.0)*cos(b*pi / 180.0);
	temp.y = length*sin(a*pi / 180.0)*sin(b*pi / 180.0);
	temp.z = length*(1 - cos(a*pi / 180.0));
}

// ----------------------------------------------------------------------------
/*---------------------------------------------------------------------
函数名：getPoints
输入：GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height

radius：底面圆的半径；
slices：从径向分，底面圆分成slices等分；侧边也分成slices等分；
circle：从切向分，底面圆形和侧边部分球面分为circle等分

bottom：底面圆位于整个球体的位置，单位为角度
height：侧面上边缘位于整个球体的位置，单位为角度。故height>bottom

输出：std::vector<point> &points

功能：获得模型的顶点坐标
---------------------------------------------------------------------*/
void getPoints(std::vector<point> &points, GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height)
{
	int i, j, w = 2 * circle, h = slices;
	float a = 0.0, b = 0.0;

	float hStep = (height - bottom) / h;	//侧边顶点的角度步长
	float bStep = radius / slices;		//底面顶点的步长
	float wStep = 360.0 / w;		//绘制所有的顶点

	point temp;
	temp.x = 0; temp.y = 0; temp.z = 0;		//以原点为第一个点
	//points.push_back(temp);

	for (a = bStep, i = 0; i < h; i++, a += bStep)		//绘制底面的圆形
	{
		for (b = 0.0, j = 0; j < w; j++, b += wStep)
		{
			getPointBottom(temp, a, b);
			points.push_back(temp);
		}
	}

	float length = radius / sin(bottom*pi / 180.0);	//侧面球体的半径
	float Zcoor = length - radius / tan(bottom*pi / 180.0);

	for (a = hStep + bottom, i = 0; i < h; i++, a += hStep)		//绘制侧面，模型为球体的一部分
	{
		for (b = 0.0, j = 0; j < w; j++, b += wStep)
		{
			getPointSide(temp, length, a, b);
			temp.z = temp.z - Zcoor;
			points.push_back(temp);
		}
	}
}

void drawSlice(point &p1, point &p2, point &p3, point &p4, float i, float j, float w, float h, drawtype type)
{
	switch (type)
	{
	case SOLID:
		glBegin(GL_QUADS);
		break;
	case WIRE:
		glBegin(GL_LINE_LOOP);
		break;
	}

	glColor3f(1, 0, 0);
	glTexCoord2f(j * 1 / (w - 1), i * 1 / (h - 1)); glVertex3f(p1.x, p1.y, p1.z);
	glTexCoord2f((j + 1) * 1 / (w - 1), i * 1 / (h - 1)); glVertex3f(p2.x, p2.y, p2.z);
	glTexCoord2f((j + 1) * 1 / (w - 1), (i + 1) * 1 / (h - 1)); glVertex3f(p3.x, p3.y, p3.z);
	glTexCoord2f(j * 1 / (w - 1), (i + 1) * 1 / (h - 1)); glVertex3f(p4.x, p4.y, p4.z);
	glEnd();

}

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type)
{
	std::vector<point> points;
	float i = 0, j = 0, w = 2 * circle, h = 2 * slices;

	getPoints(points, radius, slices, circle, bottom, height);

	if (points.empty())
	{
		return 0;
	}

	for (; i < h - 1; i++)
	{
		for (j = 0; j < w - 1; j++)
			drawSlice(points[(int)(i*w + j)], points[(int)(i*w + j + 1)], points[(int)((i + 1)*w + j + 1)], points[(int)((i + 1)*w + j)], i, j, w, h, type);
		drawSlice(points[(int)(i*w + j)], points[(int)(i*w)], points[(int)((i + 1)*w)], points[(int)((i + 1)*w + j)], i, j, w, h, type);	//绘制循环中最后一个点到第一个点的图像
	}

	points.clear();
	return 1;
}