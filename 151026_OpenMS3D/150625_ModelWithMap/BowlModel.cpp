

#include "BowlModel.h"

#include <vector>

const double pi = 3.1415926;

struct Point3f		//球体模型顶点
{
	GLfloat x;
	GLfloat y;
	GLfloat z;
};

struct Point2f		//相机成像的像素坐标
{
	GLfloat x;
	GLfloat y;
};

void getPointBottom(Point3f& temp, float a, float b)
{
	temp.x = a*cos(b*pi / 180.0);
	temp.y = a*sin(b*pi / 180.0);
	temp.z = 0;
}

void getPointSide(Point3f& temp, GLfloat length, float a, float b)
{
	temp.x = length*sin(a*pi / 180.0)*cos(b*pi / 180.0);
	temp.y = length*sin(a*pi / 180.0)*sin(b*pi / 180.0);
	temp.z = length*(1 - cos(a*pi / 180.0));
}

void initTexture(const char* filename1, const char* filename2, const char* filename3, const char* filename4)
{
	// 图片反转，使像素坐标原点位于左下角
	Front = cv::imread(filename1);
	cv::flip(Front, FrontMap, 0);

	Left = cv::imread(filename2);
	cv::flip(Left, LeftMap, 0);

	Right = cv::imread(filename3);
	cv::flip(Right, RightMap, 0);

	Rear = cv::imread(filename4);
	cv::flip(Rear, RearMap, 0);

	glGenTextures(4, texttureID);

	//	初始化贴图：前视图
	glBindTexture(GL_TEXTURE_2D, texttureID[0]);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FrontMap.cols, FrontMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, FrontMap.data);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	//	初始化贴图：左视图
	glBindTexture(GL_TEXTURE_2D, texttureID[1]);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, LeftMap.cols, LeftMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, LeftMap.data);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	//	初始化贴图：右视图
	glBindTexture(GL_TEXTURE_2D, texttureID[2]);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RightMap.cols, RightMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, RightMap.data);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	//	初始化贴图：后视图
	glBindTexture(GL_TEXTURE_2D, texttureID[3]);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RearMap.cols, RearMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, RearMap.data);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

// ----------------------------------------------------------------------------
/*---------------------------------------------------------------------
函数名：getPoints
输入：GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height

radius：底面圆的半径；
slices：从径向分，底面圆分成slices等分；侧边也分成slices等分；
circle：从切向分，底面圆形和侧边部分球面分为circle等分

bottom：底面圆位于整个球体的位置，单位为角度
height：侧面上边缘位于整个球体的位置，单位为角度。故height>bottom(没有在程序中检查！)

输出：std::vector<point> &points

功能：获得模型的顶点坐标
---------------------------------------------------------------------*/
void getPoints(std::vector<Point3f> &points, GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height)
{
	int i, j, w = 2 * circle, h = slices;
	float a = 0.0, b = 0.0;

	float hStep = (height - bottom) / h;	//侧边顶点的角度步长
	float bStep = radius / slices;		//底面顶点的步长
	float wStep = 360.0 / w;		//绘制所有的顶点

	Point3f temp;

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

int chooseTextureID(float j, float w)
{
	j = (int)j;		w = (int)w;

	int temp = j / (w / 4);

	switch (temp)
	{
	case 0:	return 0;
	case 1:	return 1;
	case 2:	return 2;
	case 3:	return 3;
	default:	return -1;
		break;
	}
}

void drawSlice(Point3f &p1, Point3f &p2, Point3f &p3, Point3f &p4, float i, float j, float w, float h, drawtype type)
{
	int ID = chooseTextureID(j, w);
	//	选择纹理图片
	glBindTexture(GL_TEXTURE_2D, texttureID[ID]);
	//glBindTexture(GL_TEXTURE_2D, texttureID[0]);

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
	glTexCoord2f((j - ID*w / 4) * 1 / (w / 4 - 1), i * 1 / (h - 1)); glVertex3f(p1.x, p1.y, p1.z);
	glTexCoord2f((j - ID*w / 4 + 1) * 1 / (w / 4 - 1), i * 1 / (h - 1)); glVertex3f(p2.x, p2.y, p2.z);
	glTexCoord2f((j - ID*w / 4 + 1) * 1 / (w / 4 - 1), (i + 1) * 1 / (h - 1)); glVertex3f(p3.x, p3.y, p3.z);
	glTexCoord2f((j - ID*w / 4) * 1 / (w / 4 - 1), (i + 1) * 1 / (h - 1)); glVertex3f(p4.x, p4.y, p4.z);
	glEnd();

}

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type)
{
	std::vector<Point3f> points;
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