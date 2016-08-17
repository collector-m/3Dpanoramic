#include "Sphere.h"
#include "GL/glut.h"

#include <opencv2/highgui/highgui.hpp>

#define pi 3.1415926
#define SOLID 3000
#define WIRE  3001

IplImage* pImage;

typedef int SPHERE_MODE;

typedef struct Point3f		//球体模型顶点
{
	GLfloat x;
	GLfloat y;
	GLfloat z;
}point;

typedef struct Point2f		//贴图坐标点
{
	GLfloat x;
	GLfloat y;
}pointMap;

void init(void);
void reshapeSphere(int w, int h);
void displaySphere(void);
int getPoint(GLfloat radius, GLfloat a, GLfloat b, point &p);
void drawSlice(point &p1, point &p2, point &p3, point &p4, float i, float j, float w, float h, SPHERE_MODE mode);
point* getPointMatrix(GLfloat radius, GLint slices);
int drawSphere(GLfloat radius, GLint slices, SPHERE_MODE mode);

int main()
{
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Sphere");
	init();
	glutReshapeFunc(reshapeSphere);
	glutDisplayFunc(displaySphere);

	pImage = cvLoadImage("earth.jpg");
	cvShowImage("Earth", pImage);

	GLuint texttureID;
	glGenTextures(1, &texttureID);
	glBindTexture(GL_TEXTURE_2D, texttureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pImage->width, pImage->height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, pImage->imageData);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glutMainLoop();

	cvReleaseImage(&pImage);

	return 0;
}
void init(void)
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClearDepth(1);
	glShadeModel(GL_SMOOTH);
	GLfloat _ambient[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat _diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat _specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat _position[] = { 200, 200, 200, 0 };
	glLightfv(GL_LIGHT0, GL_AMBIENT, _ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, _diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, _specular);
	glLightfv(GL_LIGHT0, GL_POSITION, _position);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}
void reshapeSphere(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 500, 0.0, 500, -500, 500);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void displaySphere(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glTranslated(250, 250, 0);
	glRotated(30, 1, 0, 0);
	glRotated(60, 0, 1, 0);
	glRotated(90, 0, 0, 1);
	glColor3f(1.0, 1.0, 1.0);
	drawSphere(200, 20, SOLID);
	glFlush();
}
int getPoint(GLfloat radius, GLfloat a, GLfloat b, point &p)
{
	p.x = radius*sin(a*pi / 180.0)*cos(b*pi / 180.0);
	p.y = radius*sin(a*pi / 180.0)*sin(b*pi / 180.0);
	p.z = radius*cos(a*pi / 180.0);
	return 1;
}
void drawSlice(point &p1, point &p2, point &p3, point &p4, float i, float j, float w, float h, SPHERE_MODE mode)
{
	switch (mode)
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

	//glColor3f(1, 0, 0);
	//glTexCoord2f(0, 0); glVertex3f(p1.x, p1.y, p1.z);
	//glTexCoord2f(0, 1.0f); glVertex3f(p2.x, p2.y, p2.z);
	//glTexCoord2f(1.0f, 1.0f); glVertex3f(p3.x, p3.y, p3.z);
	//glTexCoord2f(1.0f, 0); glVertex3f(p4.x, p4.y, p4.z);
	//glEnd();
}
point* getPointMatrix(GLfloat radius, GLint slices)
{
	int i, j, w = 2 * slices, h = slices;
	float a = 0.0, b = 0.0;
	float hStep = 180.0 / (h-1);
	float wStep = 360.0 / w;
	int length = w*h;
	point *matrix;
	matrix = (point *)malloc(length*sizeof(point));
	if (!matrix)return NULL;
	for (a = 0.0, i = 0; i<h; i++, a += hStep)
		for (b = 0.0, j = 0; j<w; j++, b += wStep)
			getPoint(radius, a, b, matrix[i*w + j]);
	return matrix;
}
int drawSphere(GLfloat radius, GLint slices, SPHERE_MODE mode)
{
	float i = 0, j = 0, w = 2 * slices, h = slices;
	point *mx;
	mx = getPointMatrix(radius, slices);
	if (!mx)return 0;
	for (; i<h - 1; i++)
	{
		for (j = 0; j<w - 1; j++)
			drawSlice(mx[(int)(i*w + j)], mx[(int)(i*w + j + 1)], mx[(int)((i + 1)*w + j + 1)], mx[(int)((i + 1)*w + j)], i, j, w, h, mode);
		drawSlice(mx[(int)(i*w + j)], mx[(int)(i*w)], mx[(int)((i + 1)*w)], mx[(int)((i + 1)*w + j)], i, j, w, h, mode);
	}
	free(mx);
	return 1;
}