/*---------------------------------------------------------------------
2015/06/24: MouseEvent		by: 胡诗卉

内容：绘制碗状模型, 并贴图;

基于：OpenGL开源库
---------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <math.h>

#include <GL/glut.h>

#include "opencv2/opencv.hpp"

#include <IL/il.h>

// assimp include files. These three are usually needed.
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <string>
#include <map>
#include <fstream>
#include <iostream>

#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)

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

//	纹理ID
GLuint texttureID[4];

//	鼠标前一时刻位置
static int xpos = 0, ypos = 0;

static bool isPressLeftButton = false, isPressRightButton = false, isPressMiddleButton = false;

// 鼠标操作的全局变量
static GLfloat  rot_x = 0, rot_y = 0, rot = 0;	//	旋转
static GLfloat  tra_x = 0, tra_y = 0, tra_z = 0;	//	平移
static GLfloat  scale = 1;	//	尺度
static GLfloat distance = 1;	//	距离

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

// the global Assimp scene object
const struct aiScene* scene = NULL;
GLuint scene_list = 0;
aiVector3D scene_min, scene_max, scene_center;

// Create an instance of the Importer class
Assimp::Importer importer;

//The default model path
//static std::string modelpath = "D:\\Program Files\\Assimp\\models\\X\\jeep1.ms3d";
static std::string modelpath = "F:\\Graphical_extend\\Assimp\\models\\OBJ\\car1.obj";
//static std::string modelpath = "C:\\Users\\Administrator\\Desktop\\assimp-3.1.1\\assimp-3.1.1\\test\\models-nonbsd\\X\\jeep1.ms3d";

// images / texture
std::map<std::string, GLuint*> textureIdMap;	// map image filenames to textureIds
GLuint*	textureIds;							// pointer to texture Array

enum drawtype	//模型绘制类型（SOLID:方格；WIRE:线型）
{
	SOLID,
	WIRE
};

/*---------------------------------------------------------------------

	鼠标响应函数：

---------------------------------------------------------------------*/
void PressLeftButton(int x, int y)
{
	rot_x += (y - ypos) / 1000.0;
	rot_y += (x - xpos) / 1000.0;

	//rot += sqrt((x - xpos)*(x - xpos) + (y - ypos)*(y - ypos)) / 500.0;
}

void PressRightButton(int x, int y)
{
	tra_x += (x - xpos) / 500.0;
	tra_y += (y - ypos) / 500.0;
}

void PressMiddleButton(int x, int y)
{
	distance += (y - ypos) / 1000.0;
}

void moveMouse(int x, int y)
{
	if (isPressLeftButton)
		PressLeftButton(x, y);
	if (isPressRightButton)
		PressRightButton(x, y);
	if (isPressMiddleButton)
		PressMiddleButton(x, y);
}

//	点击鼠标响应函数
void pressMouse(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN)
		{
			xpos = x;
			ypos = y;
			isPressLeftButton = true;
		}

		if (state == GLUT_UP)
			isPressLeftButton = false;
		break;

	case GLUT_MIDDLE_BUTTON:
		if (state == GLUT_DOWN)
		{
			xpos = x;
			ypos = y;
			isPressMiddleButton = true;
		}

		if (state == GLUT_UP)
			isPressMiddleButton = false;
		break;
		break;

	case GLUT_RIGHT_BUTTON:
		if (state == GLUT_DOWN)
		{
			xpos = x;
			ypos = y;
			isPressRightButton = true;
		}

		if (state == GLUT_UP)
			isPressRightButton = false;
		break;

	default:
		break;
	}
}


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
height：侧面上边缘位于整个球体的位置，单位为角度。故height>bottom

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

// ----------------------------------------------------------------------------
void get_bounding_box_for_node(const aiNode* nd,
	aiVector3D* min,
	aiVector3D* max,
	aiMatrix4x4* trafo)
{
	aiMatrix4x4 prev;
	unsigned int n = 0, t;

	prev = *trafo;
	//aiMultiplyMatrix4(trafo, &nd->mTransformation);

	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t) {

			aiVector3D tmp = mesh->mVertices[t];
			//aiTransformVecByMatrix4(&tmp, trafo);

			min->x = aisgl_min(min->x, tmp.x);
			min->y = aisgl_min(min->y, tmp.y);
			min->z = aisgl_min(min->z, tmp.z);

			max->x = aisgl_max(max->x, tmp.x);
			max->y = aisgl_max(max->y, tmp.y);
			max->z = aisgl_max(max->z, tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		get_bounding_box_for_node(nd->mChildren[n], min, max, trafo);
	}
	*trafo = prev;
}

// ----------------------------------------------------------------------------
void get_bounding_box(aiVector3D* min, aiVector3D* max)
{
	aiMatrix4x4 trafo;
	aiIdentityMatrix4(&trafo);

	min->x = min->y = min->z = 1e10f;
	max->x = max->y = max->z = -1e10f;
	get_bounding_box_for_node(scene->mRootNode, min, max, &trafo);
}


void resharpeModel(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//	正交投影
	//glOrtho(-400.0, 400.0, -400.0, 400.0, -400.0, 400.0);
	//	透视投影
	gluPerspective(100.0f, (GLfloat)w / (GLfloat)h, 0.1f, 600.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

// ----------------------------------------------------------------------------
bool ImportMs3dFile(const std::string& pFile)
{
	//check if file exists
	std::ifstream fin(pFile.c_str());
	if (!fin.fail())
	{
		fin.close();
	}
	else
	{
		std::cout << "The Model path is not exist!" << std::endl;
		return false;
	}

	scene = importer.ReadFile(pFile, aiProcessPreset_TargetRealtime_Quality);

	if (!scene)
	{
		std::cout << "Import Model failed!" << std::endl;
		return false;
	}

	get_bounding_box(&scene_min, &scene_max);
	scene_center.x = (scene_min.x + scene_max.x) / 2.0f;
	scene_center.y = (scene_min.y + scene_max.y) / 2.0f;
	scene_center.z = (scene_min.z + scene_max.z) / 2.0f;

	std::cout << "Import Model successfully!!..." << std::endl;
	return true;
}

// ----------------------------------------------------------------------------
std::string getBasePath(const std::string& path)
{
	size_t pos = path.find_last_of("\\/");
	return (std::string::npos == pos) ? "" : path.substr(0, pos + 1);
}

// ----------------------------------------------------------------------------
bool LoadGLTextures(const aiScene* scene)
{
	ILboolean success;

	/* Before calling ilInit() version should be checked. */
	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION)
	{
		std::cout << "Wrong DevIL version. Old devil.dll in system32/SysWow64?" << std::endl;
		return -1;
	}

	ilInit(); /* Initialization of DevIL */

	if (scene->HasTextures())
	{
		std::cout << "Support for meshes with embedded textures is not implemented" << std::endl;
		return -1;
	}

	/* getTexture Filenames and Numb of Textures */
	for (unsigned int m = 0; m<scene->mNumMaterials; m++)
	{
		int texIndex = 0;
		aiReturn texFound = AI_SUCCESS;

		aiString path;	// filename

		while (texFound == AI_SUCCESS)
		{
			texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
			textureIdMap[path.data] = NULL; //fill map with textures, pointers still NULL yet
			texIndex++;
		}
	}

	int numTextures = textureIdMap.size();

	/* array with DevIL image IDs */
	ILuint* imageIds = NULL;
	imageIds = new ILuint[numTextures];

	/* generate DevIL Image IDs */
	ilGenImages(numTextures, imageIds); /* Generation of numTextures image names */

	/* create and fill array with GL texture ids */
	textureIds = new GLuint[numTextures];
	glGenTextures(numTextures, textureIds); /* Texture name generation */

	/* get iterator */
	std::map<std::string, GLuint*>::iterator itr = textureIdMap.begin();

	std::string basepath = getBasePath(modelpath);
	for (int i = 0; i<numTextures; i++)
	{

		//save IL image ID
		std::string filename = (*itr).first;  // get filename
		(*itr).second = &textureIds[i];	  // save texture id for filename in map
		itr++;								  // next texture


		ilBindImage(imageIds[i]); /* Binding of DevIL image name */
		std::string fileloc = basepath + filename;	/* Loading of image */
		success = ilLoadImage((const wchar_t*)fileloc.c_str());

		ILenum test = ilGetInteger(IL_IMAGE_FORMAT);

		if (success) /* If no error occured: */
		{
			success = ilConvertImage(IL_RGB, IL_UNSIGNED_BYTE); /* Convert every colour component into
																unsigned byte. If your image contains alpha channel you can replace IL_RGB with IL_RGBA */
			if (!success)
			{
				/* Error occured */
				std::cout << "Couldn't convert image" << std::endl;
				return -1;
			}

			/*---------------使用静态纹理--------------------*/

			//glGenTextures(numTextures, &textureIds[i]); /* Texture name generation */
			glBindTexture(GL_TEXTURE_2D, textureIds[i]); /* Binding of texture name */
			//redefine standard texture values
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); /* We will use linear
																			  interpolation for magnification filter */
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); /* We will use linear
																			  interpolation for minifying filter */

			ILenum test = ilGetInteger(IL_IMAGE_FORMAT);
			glTexImage2D(GL_TEXTURE_2D, 0, ilGetInteger(IL_IMAGE_BPP), ilGetInteger(IL_IMAGE_WIDTH),
				ilGetInteger(IL_IMAGE_HEIGHT), 0, ilGetInteger(IL_IMAGE_FORMAT), GL_UNSIGNED_BYTE,
				ilGetData()); /* Texture specification */

		}
		else
		{
			/* Error occured */
			std::cout << "Couldn't load Image" << std::endl;
			return -1;
		}
	}

	ilDeleteImages(numTextures, imageIds); /* Because we have already copied image data into texture data
										   we can release memory used by image. */

	//Cleanup
	delete[] imageIds;
	imageIds = NULL;

	//return success;
	return true;
}

// ----------------------------------------------------------------------------
void set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

// ----------------------------------------------------------------------------
void color4_to_float4(const aiColor4D *c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

// ----------------------------------------------------------------------------
void apply_material(const struct aiMaterial* mtl)
{
	float c[4];

	GLenum fill_mode;
	int ret1, ret2;
	aiColor4D diffuse;
	aiColor4D specular;
	aiColor4D ambient;
	aiColor4D emission;
	float shininess, strength;
	int two_sided;
	int wireframe;
	unsigned int max;	// changed: to unsigned

	int texIndex = 0;
	aiString texPath;	//contains filename of texture

	if (AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, texIndex, &texPath))
	{
		//bind texture
		unsigned int texID = *textureIdMap[texPath.data];
		glBindTexture(GL_TEXTURE_2D, texID);
	}

	set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
		color4_to_float4(&diffuse, c);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, c);

	set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
		color4_to_float4(&specular, c);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);

	set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
		color4_to_float4(&ambient, c);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, c);

	set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
		color4_to_float4(&emission, c);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, c);

	max = 1;
	ret1 = aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
	max = 1;
	ret2 = aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS_STRENGTH, &strength, &max);
	if ((ret1 == AI_SUCCESS) && (ret2 == AI_SUCCESS))
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess * strength);
	else {
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0f);
		set_float4(c, 0.0f, 0.0f, 0.0f, 0.0f);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);
	}

	max = 1;
	if (AI_SUCCESS == aiGetMaterialIntegerArray(mtl, AI_MATKEY_ENABLE_WIREFRAME, &wireframe, &max))
		fill_mode = wireframe ? GL_LINE : GL_FILL;
	else
		fill_mode = GL_FILL;
	glPolygonMode(GL_FRONT_AND_BACK, fill_mode);

	max = 1;
	if ((AI_SUCCESS == aiGetMaterialIntegerArray(mtl, AI_MATKEY_TWOSIDED, &two_sided, &max)) && two_sided)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);
}

// ----------------------------------------------------------------------------
void recursive_render(const struct aiScene* sc, const struct aiNode* nd)
{
	unsigned int i, n, t;
	aiMatrix4x4 m = nd->mTransformation;

	// update transform
	aiTransposeMatrix4(&m);
	glPushMatrix();
	glMultMatrixf((float*)&m);

	// draw all meshes assigned to this node
	for (n = 0; n < nd->mNumMeshes; ++n)
	{
		const struct aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];

		//apply_material(sc->mMaterials[mesh->mMaterialIndex]);

		if (mesh->mNormals == NULL)
		{
			glDisable(GL_LIGHTING);
		}
		else
			glEnable(GL_LIGHTING);

		for (t = 0; t < mesh->mNumFaces; ++t)
		{
			const struct aiFace* face = &mesh->mFaces[t];
			GLenum face_mode;

			switch (face->mNumIndices)
			{
			case 1: face_mode = GL_POINTS; break;
			case 2: face_mode = GL_LINES; break;
			case 3: face_mode = GL_TRIANGLES; break;
			default: face_mode = GL_POLYGON; break;
			}

			glBegin(GL_TRIANGLES);

			for (i = 0; i < face->mNumIndices; ++i)
			{
				int index = face->mIndices[i];
				if (mesh->mColors[0] != NULL)
					glColor4fv((GLfloat*)&mesh->mColors[0][index]);
				if (mesh->mNormals != NULL)
					glNormal3fv(&mesh->mNormals[index].x);
				if (mesh->HasTextureCoords(0))		//HasTextureCoords(texture_coordinates_set)
				{
					glTexCoord2f(mesh->mTextureCoords[0][index].x, 1 - mesh->mTextureCoords[0][index].y); //mTextureCoords[channel][vertex]
				}
				glVertex3fv(&mesh->mVertices[index].x);
			}

			glEnd();
		}
	}

	// draw all children
	for (n = 0; n < nd->mNumChildren; ++n)
	{
		recursive_render(sc, nd->mChildren[n]);
	}

	glPopMatrix();
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
	float tmp = 0.08;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.f, 1.f, 5.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f);

	if (scene_list == 0)
	{
		//scene_list = glGenLists(1);
		glNewList(scene_list, GL_COMPILE);
		// now begin at the root node of the imported data and traverse
		// the scenegraph by multiplying subsequent local transforms
		// together on GL's matrix stack.
		
		//	鼠标操作
		glRotatef(30 * rot_x, 1.f, 0.f, 0.f);
		glRotatef(30 * rot_y, 0.f, 1.f, 0.f);

		glScalef(tmp, tmp, tmp);

		recursive_render(scene, scene->mRootNode);
		glTranslatef(0, -scene_max.y, 0);
		glRotatef(-90, 1.f, 0.f, 0.f);
		glScalef(0.1, 0.1, 0.1);
		drawModel(300, 5, 20, 30, 60, SOLID);
		glEndList();
	}

	 //图片反转，使像素坐标原点位于左下角
	Video >> Front;
	cv::flip(Front, FrontMap, 0);

	//Video >> LeftMap;
	//cv::flip(Left, LeftMap, 0);

	//Video >> RightMap;
	//cv::flip(Right, RightMap, 0);

	//Video >> RearMap;
	//cv::flip(Rear, RearMap, 0);

	//Front = cv::imread("front.bmp");
	//cv::flip(Front, FrontMap, 0);

	//Left = cv::imread("left.bmp");
	//cv::flip(Left, LeftMap, 0);

	//Right = cv::imread("Right.bmp");
	//cv::flip(Right, RightMap, 0);

	//Rear = cv::imread("rear.bmp");
	//cv::flip(Rear, RearMap, 0);

	//glGenTextures(4, texttureID);
	//	初始化贴图：前视图
	glBindTexture(GL_TEXTURE_2D, texttureID[0]);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FrontMap.cols, FrontMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, FrontMap.data);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

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

	glCallList(scene_list);
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

	glutMouseFunc(pressMouse);
	glutMotionFunc(moveMouse);

	if (!ImportMs3dFile(modelpath)) return -1;

	glEnable(GL_TEXTURE_2D);
	//
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);

	//glEnable(GL_DEPTH_TEST);

	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	//
	GLfloat LightAmbient[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat LightPosition[] = { 0.0f, 0.0f, 15.0f, 1.0f };

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);    // Uses default lighting parameters
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);

	glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, LightPosition);
	glEnable(GL_LIGHT1);

	glEnable(GL_NORMALIZE);

	//// XXX docs say all polygons are emitted CCW, but tests show that some aren't.
	//if (getenv("MODEL_IS_BROKEN"))
	//	glFrontFace(GL_CW);

	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);


	// 图片反转，使像素坐标原点位于左下角
	//Front = cv::imread("front.bmp");
	//cv::flip(Front, FrontMap, 0);

	Left = cv::imread("left.bmp");
	cv::flip(Left, LeftMap, 0);

	Right = cv::imread("Right.bmp");
	cv::flip(Right, RightMap, 0);

	Rear = cv::imread("rear.bmp");
	cv::flip(Rear, RearMap, 0);


	glutGet(GLUT_ELAPSED_TIME);

	glGenTextures(4, texttureID);

	////	初始化贴图：前视图
	//glBindTexture(GL_TEXTURE_2D, texttureID[0]);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FrontMap.cols, FrontMap.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, FrontMap.data);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

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

	// cleanup - calling 'aiReleaseImport' is important, as the library 
	// keeps internal resources until the scene is freed again. Not 
	// doing so can cause severe resource leaking.
	aiReleaseImport(scene);

	// We added a log stream to the library, it's our job to disable it
	// again. This will definitely release the last resources allocated
	// by Assimp.
	aiDetachAllLogStreams();

	return 0;
}