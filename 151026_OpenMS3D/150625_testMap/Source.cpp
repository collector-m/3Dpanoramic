/*---------------------------------------------------------------------
2015/06/24: testMap		by: 胡诗卉

内容：绘制碗状模型, 并贴图;

基于：OpenGL开源库
---------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>

#include <vector>

#include <GL/glut.h>

#include "opencv2/opencv.hpp"

const double pi = 3.1415926;

//	纹理ID
GLuint texttureID[4];

// Rotation
static float angle = 0.f;

GLfloat LightAmbient[] = { 0.5f, 0.5f, 0.5f, 1.0f };
GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat LightPosition[] = { 0.0f, 0.0f, 15.0f, 1.0f };

//	打开摄像头
cv::VideoCapture Video(0);

//	贴图图片
cv::Mat Front, Left, Right, Rear;
cv::Mat FrontMap, LeftMap, RightMap, RearMap;

enum drawtype	//模型绘制类型（SOLID:方格；WIRE:线型）
{
	SOLID,
	WIRE
};

typedef struct Point3f		//球体模型顶点
{
	GLfloat x;
	GLfloat y;
	GLfloat z;
}point;

void init(void)		//glut初始化
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClearDepth(1);
	glShadeModel(GL_SMOOTH);

	//gluLookAt(0.f, 1.f, 3.f, 0.f, 0.f, -5.f, 0.f, 1.f, 0.f);

	GLfloat _ambient[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat _diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat _specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat _position[] = { 0, 200, 0, 0 };
	glLightfv(GL_LIGHT0, GL_AMBIENT, _ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, _diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, _specular);
	glLightfv(GL_LIGHT0, GL_POSITION, _position);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

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

void drawSlice(point &p1, point &p2, point &p3, point &p4, float i, float j, float w, float h, drawtype type)
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
	glTexCoord2f((j - ID*w/4) * 1 / (w / 4 - 1), i * 1 / (h - 1)); glVertex3f(p1.x, p1.y, p1.z);
	glTexCoord2f((j - ID*w / 4 + 1) * 1 / (w / 4 - 1), i * 1 / (h - 1)); glVertex3f(p2.x, p2.y, p2.z);
	glTexCoord2f((j - ID*w / 4 + 1) * 1 / (w / 4 - 1), (i + 1) * 1 / (h - 1)); glVertex3f(p3.x, p3.y, p3.z);
	glTexCoord2f((j - ID*w / 4) * 1 / (w / 4 - 1), (i + 1) * 1 / (h - 1)); glVertex3f(p4.x, p4.y, p4.z);
	glEnd();

}

int drawModel(GLfloat radius, GLint slices, GLint circle, GLfloat bottom, GLfloat height, drawtype type)
{
	std::vector<point> points;
	float i = 0, j = 0, w = 2 * circle, h = 2 * slices;

	//	时间开销可以放在循环外
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


void resharpeModel(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//	正交投影
	//glOrtho(-400.0, 400.0, -400.0, 400.0, -400.0, 400.0);
	//	透视投影
	gluPerspective(110.0f, (GLfloat)w / (GLfloat)h, 0.1f, 600.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

// ----------------------------------------------------------------------------
void do_motion(void)
{
	static GLint prev_time = 0;
	static GLint prev_fps_time = 0;
	static int frames = 0;

	int time = glutGet(GLUT_ELAPSED_TIME);
	angle += (time - prev_time)*0.01;
	prev_time = time;

	++frames;
	if ((time - prev_fps_time) > 1000)
	{
		int current_fps = frames * 1000 / (time - prev_fps_time);
		printf("%d fps\n", current_fps);
		frames = 0;
		prev_fps_time = time;
	}

	glutPostRedisplay();
}

void displayModel()
{
	float tmp = 1;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//gluLookAt(0.f, -1.f, 3.f, 0.f, 0.f, -1.f, 0.f, 1.f, 0.f);
	//	正交投影
	gluLookAt(0.f, -100.f, -300.f, 0.f, 0.f, -100.f, 0.f, 1.f, 0.f);
	//	透视投影
	gluLookAt(0.f, 100.f, 300.f, 0.f, 0.f, -300.f, 0.f, 1.f, 0.f);

	//glTranslated(0, 200, 500.f);
	glRotatef(-90, 1.f, 0.f, 0.f);

	// rotate it around the z axis
	glRotatef(angle, 0.f, 0.f, 1.f);

	// scale the whole asset to fit into our view frustum 
	glScalef(tmp, tmp, tmp);

	//glRotated(30, 1, 0, 0);
	//glRotated(60, 0, 1, 0);
	//glRotated(90, 0, 0, 1);
	glColor3f(1.0, 1.0, 1.0);

	// 图片反转，使像素坐标原点位于左下角
	//Video >> Front;
	//cv::flip(Front, FrontMap, 0);

	//Video >> LeftMap;
	//cv::flip(Left, LeftMap, 0);

	//Video >> RightMap;
	//cv::flip(Right, RightMap, 0);

	//Video >> RearMap;
	//cv::flip(Rear, RearMap, 0);

	////	初始化贴图：前视图
	//glBindTexture(GL_TEXTURE_2D, texttureID[0]);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FrontMap.cols, FrontMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, FrontMap.data);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	////	初始化贴图：左视图
	//glBindTexture(GL_TEXTURE_2D, texttureID[1]);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, LeftMap.cols, LeftMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, LeftMap.data);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	////	初始化贴图：右视图
	//glBindTexture(GL_TEXTURE_2D, texttureID[2]);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RightMap.cols, RightMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, RightMap.data);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	////	初始化贴图：后视图
	//glBindTexture(GL_TEXTURE_2D, texttureID[3]);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RearMap.cols, RearMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, RearMap.data);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	//cv::imshow("Camera", Front);

	drawModel(300, 5, 20, 30, 60, SOLID);

	glutSwapBuffers();
	do_motion();
	glFlush();
}

int main()
{
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("testModel");


	// Opengl 初始化设置
	glEnable(GL_TEXTURE_2D);

	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();

	init();
	glutReshapeFunc(resharpeModel);
	glutDisplayFunc(displayModel);

	// 图片反转，使像素坐标原点位于左下角
	Front = cv::imread("front.bmp");
	cv::flip(Front, FrontMap, 0);

	Left = cv::imread("left.bmp");
	cv::flip(Left, LeftMap, 0);

	Right = cv::imread("Right.bmp");
	cv::flip(Right, RightMap, 0);

	Rear = cv::imread("rear.bmp");
	cv::flip(Rear, RearMap, 0);

	//Video >> Front;
	//cv::flip(Front, FrontMap, 0);

	//Video >> Left;
	//cv::flip(Left, LeftMap, 0);

	//Video >> Right;
	//cv::flip(Right, RightMap, 0);

	//Video >> Rear;
	//cv::flip(Rear, RearMap, 0);

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

	glutMainLoop();

	return 0;
}