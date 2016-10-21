

#include "BowlModel.h"

#include <vector>

const double pi = 3.1415926;

typedef struct Point3f		//����ģ�Ͷ���
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
��������getPoints
���룺GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height

radius������Բ�İ뾶��
slices���Ӿ���֣�����Բ�ֳ�slices�ȷ֣����Ҳ�ֳ�slices�ȷ֣�
circle��������֣�����Բ�κͲ�߲��������Ϊcircle�ȷ�

bottom������Բλ�����������λ�ã���λΪ�Ƕ�
height�������ϱ�Եλ�����������λ�ã���λΪ�Ƕȡ���height>bottom

�����std::vector<point> &points

���ܣ����ģ�͵Ķ�������
---------------------------------------------------------------------*/
void getPoints(std::vector<point> &points, GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height)
{
	int i, j, w = 2 * circle, h = slices;
	float a = 0.0, b = 0.0;

	float hStep = (height - bottom) / h;	//��߶���ĽǶȲ���
	float bStep = radius / slices;		//���涥��Ĳ���
	float wStep = 360.0 / w;		//�������еĶ���

	point temp;
	temp.x = 0; temp.y = 0; temp.z = 0;		//��ԭ��Ϊ��һ����
	//points.push_back(temp);

	for (a = bStep, i = 0; i < h; i++, a += bStep)		//���Ƶ����Բ��
	{
		for (b = 0.0, j = 0; j < w; j++, b += wStep)
		{
			getPointBottom(temp, a, b);
			points.push_back(temp);
		}
	}

	float length = radius / sin(bottom*pi / 180.0);	//��������İ뾶
	float Zcoor = length - radius / tan(bottom*pi / 180.0);

	for (a = hStep + bottom, i = 0; i < h; i++, a += hStep)		//���Ʋ��棬ģ��Ϊ�����һ����
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
		drawSlice(points[(int)(i*w + j)], points[(int)(i*w)], points[(int)((i + 1)*w)], points[(int)((i + 1)*w + j)], i, j, w, h, type);	//����ѭ�������һ���㵽��һ�����ͼ��
	}

	points.clear();
	return 1;
}