/*------------------------------------------------
	testLookAt:	≤‚ ‘ ”Ω«Œª÷√
	
------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>

#include <GL/glut.h>

#include "opencv2/opencv.hpp"
#include <iostream>

// Rotation
static float angle = 0.f;

CvCapture* pCapture;
IplImage* pTest;

GLuint TextureID;

// ----------------------------------------------------------------------------
void reshape(int width, int height)
{
	const double aspectRatio = (float)width / height, fieldOfView = 45.0;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, aspectRatio,
		1.0, 1000.0);  /* Znear and Zfar */
	glViewport(0, 0, width, height);

	glutSwapBuffers();
}

// ----------------------------------------------------------------------------
void do_motion(void)
{
	static GLint prev_time = 0;
	static GLint prev_fps_time = 0;
	static int frames = 0;

	int time = glutGet(GLUT_ELAPSED_TIME);
	angle += (time - prev_time)*0.05;
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

// ----------------------------------------------------------------------------
void display(void)
{
	float tmp = 0.5;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.f, 0.f, 5.f, -1.f, 0.f, 0.f, 1.f, 0.f, 0.f);

	// rotate it around the y axis
	glRotatef(angle, 0.f, 1.f, 0.f);

	// scale the whole asset to fit into our view frustum 
	glScalef(tmp, tmp, tmp);

	//test = cvQueryFrame(pCapture);

	//if (!test) {
	//	std::cout << "◊•»°…„œÒÕ∑ÕºœÒ¥ÌŒÛ£°" << std::endl;
	//	return;
	//}

	//cvShowImage("test", test);

	//glBegin(GL_QUADS);
	//glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
	//glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
	//glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
	//glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
	//glEnd();

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, test->width, test->height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, test->imageData);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(-1.0f, -1.0f, 1.0f);
	glTexCoord2f(0, 1.0f); glVertex3f(-1.0f, 1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 1.0f, 1.0f);
	glTexCoord2f(1.0f, 0); glVertex3f(1.0f, -1.0f, 1.0f);
	glEnd();

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(1.0f, -1.0f, 1.0f);
	glTexCoord2f(0, 1.0f); glVertex3f(1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 1.0f, -1.0f);
	glTexCoord2f(1.0f, 0); glVertex3f(1.0f, 1.0f, 1.0f);
	glEnd();

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(1.0f, -1.0f, -1.0f);
	glTexCoord2f(0, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, 1.0f, -1.0f);
	glTexCoord2f(1.0f, 0); glVertex3f(1.0f, 1.0f, -1.0f);
	glEnd();

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(-1.0f, -1.0f, -1.0f);
	glTexCoord2f(0, 1.0f); glVertex3f(-1.0f, -1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, 1.0f, 1.0f);
	glTexCoord2f(1.0f, 0); glVertex3f(-1.0f, 1.0f, -1.0f);
	glEnd();

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(-1.0f, 1.0f, 1.0f);
	glTexCoord2f(0, 1.0f); glVertex3f(1.0f, 1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 1.0f, -1.0f);
	glTexCoord2f(1.0f, 0); glVertex3f(-1.0f, 1.0f, -1.0f);
	glEnd();

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(-1.0f, -1.0f, -1.0f);
	glTexCoord2f(0, 1.0f); glVertex3f(1.0f, -1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, -1.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 1.0f);
	glEnd();

	glutSwapBuffers();

	do_motion();
	glFlush();
}

// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInit(&argc, argv);

	glutCreateWindow("Assimp - Very simple OpenGL sample");
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);

	pTest = cvLoadImage("Texture.bmp");

	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, &TextureID);
	glBindTexture(GL_TEXTURE_2D, TextureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pTest->width, pTest->height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, pTest->imageData);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	//pCapture = cvCreateCameraCapture(0);
	//test = cvQueryFrame(pCapture);

	//if (!test) {
	//	std::cout << "◊•»°…„œÒÕ∑ÕºœÒ¥ÌŒÛ£°" << std::endl;
	//	return -1;
	//}

	//cvShowImage("test", test);

	glutMainLoop();

	cvReleaseImage(&pTest);
}

