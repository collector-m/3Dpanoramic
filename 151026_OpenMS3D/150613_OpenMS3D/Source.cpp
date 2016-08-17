/*---------------------------------------------------------------------
	2015/06/13: OpenMS3D			by: 胡诗卉

	内容：打开MS3D文件，并加载纹理
	基于：OpenGL、Assimp开源库
---------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>

#include <GL/glut.h>
#include <IL/il.h>

//#include "opencv2/opencv.hpp"

// assimp include files. These three are usually needed.

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <string>
#include <map>
#include <fstream>
#include <iostream>

// the global Assimp scene object
const struct aiScene* scene = NULL;
GLuint scene_list = 0;
aiVector3D scene_min, scene_max, scene_center;

// Create an instance of the Importer class
Assimp::Importer importer;

// Rotation
static float angle = 0.f;

//The default model path
//static std::string modelpath = "D:\\Program Files\\Assimp\\models\\X\\jeep1.ms3d";
static std::string modelpath = "F:\\Graphical_extend\\Assimp\\models\\OBJ\\car1.obj";

// images / texture
std::map<std::string, GLuint*> textureIdMap;	// map image filenames to textureIds
GLuint*	textureIds;							// pointer to texture Array

//// Video image
//cv::Mat image;
//cv::VideoCapture img(0);

// ----------------------------------------------------------------------------
void reshape(int width, int height)
{
	const double aspectRatio = (float)width / height, fieldOfView = 45.0;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, aspectRatio,
		1.0, 1000.0);  /* Znear and Zfar */
	glViewport(0, 0, width, height);
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
//bool LoadGLTextures(const aiScene* scene)
//{
//	ILboolean success;
//
//	/* Before calling ilInit() version should be checked. */
//	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION)
//	{
//		std::cout << "Wrong DevIL version. Old devil.dll in system32/SysWow64?" << std::endl;
//		return -1;
//	}
//
//	ilInit(); /* Initialization of DevIL */
//
//	if (scene->HasTextures())
//	{
//		std::cout << "Support for meshes with embedded textures is not implemented" << std::endl;
//		return -1;
//	}
//
//	/* getTexture Filenames and Numb of Textures */
//	for (unsigned int m = 0; m<scene->mNumMaterials; m++)
//	{
//		int texIndex = 0;
//		aiReturn texFound = AI_SUCCESS;
//
//		aiString path;	// filename
//
//		while (texFound == AI_SUCCESS)
//		{
//			texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
//			textureIdMap[path.data] = NULL; //fill map with textures, pointers still NULL yet
//			texIndex++;
//		}
//	}
//
//	int numTextures = textureIdMap.size();
//
//	/* array with DevIL image IDs */
//	ILuint* imageIds = NULL;
//	imageIds = new ILuint[numTextures];
//
//	/* generate DevIL Image IDs */
//	ilGenImages(numTextures, imageIds); /* Generation of numTextures image names */
//
//	/* create and fill array with GL texture ids */
//	textureIds = new GLuint[numTextures];
//	glGenTextures(numTextures, textureIds); /* Texture name generation */
//
//	/* define texture path */
//	//std::string texturepath = "../../../test/models/Obj/";
//
//	/* get iterator */
//	std::map<std::string, GLuint*>::iterator itr = textureIdMap.begin();
//
//	std::string basepath = getBasePath(modelpath);
//	for (int i = 0; i<numTextures; i++)
//	{
//
//		//save IL image ID
//		std::string filename = (*itr).first;  // get filename
//		(*itr).second = &textureIds[i];	  // save texture id for filename in map
//		itr++;								  // next texture
//
//
//		ilBindImage(imageIds[i]); /* Binding of DevIL image name */
//		std::string fileloc = basepath + filename;	/* Loading of image */
//		success = ilLoadImage((const wchar_t*)fileloc.c_str());
//
//		ILenum test = ilGetInteger(IL_IMAGE_FORMAT);
//
//		if (success) /* If no error occured: */
//		{
//			success = ilConvertImage(IL_RGB, IL_UNSIGNED_BYTE); /* Convert every colour component into
//																unsigned byte. If your image contains alpha channel you can replace IL_RGB with IL_RGBA */
//			if (!success)
//			{
//				/* Error occured */
//				std::cout << "Couldn't convert image" << std::endl;
//				return -1;
//			}
//
//			/*---------------使用静态纹理--------------------*/
//			
//			//glGenTextures(numTextures, &textureIds[i]); /* Texture name generation */
//			glBindTexture(GL_TEXTURE_2D, textureIds[i]); /* Binding of texture name */
//			//redefine standard texture values
//			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); /* We will use linear
//																			  interpolation for magnification filter */
//			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); /* We will use linear
//																			  interpolation for minifying filter */
//
//			ILenum test = ilGetInteger(IL_IMAGE_FORMAT);
//			glTexImage2D(GL_TEXTURE_2D, 0, ilGetInteger(IL_IMAGE_BPP), ilGetInteger(IL_IMAGE_WIDTH),
//				ilGetInteger(IL_IMAGE_HEIGHT), 0, ilGetInteger(IL_IMAGE_FORMAT), GL_UNSIGNED_BYTE,
//				ilGetData()); /* Texture specification */
//
//		}
//		else
//		{
//			/* Error occured */
//			std::cout << "Couldn't load Image" << std::endl;
//			return -1;
//		}
//	}
//
//	ilDeleteImages(numTextures, imageIds); /* Because we have already copied image data into texture data
//										   we can release memory used by image. */
//
//	//Cleanup
//	delete[] imageIds;
//	imageIds = NULL;
//
//	//return success;
//	return true;
//}

// ----------------------------------------------------------------------------
//void get_bounding_box(struct aiVector3D& min, struct aiVector3D& max)
//{
//	struct aiMatrix4x4 trafo;
//	aiIdentityMatrix4(&trafo);
//
//	min.
//}

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

		apply_material(sc->mMaterials[mesh->mMaterialIndex]);

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

// ----------------------------------------------------------------------------
void display(void)
{
	float tmp = 10;

	//img >> image;
	//cv::imshow("Camera", image);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.f, 0.f, 3.f, 0.f, 0.f, -5.f, 0.f, 1.f, 0.f);

	// rotate it around the y axis
	glRotatef(angle, 0.0f, 0.0f, 100.0f);

	// scale the whole asset to fit into our view frustum 
	glScalef(tmp, tmp, tmp);

	// center the model
	glTranslatef(-scene_center.x, -scene_center.y, -scene_center.z);

	if (scene_list == 0)
	{
		scene_list = glGenLists(1);
		glNewList(scene_list, GL_COMPILE);
		// now begin at the root node of the imported data and traverse
		// the scenegraph by multiplying subsequent local transforms
		// together on GL's matrix stack.
		recursive_render(scene, scene->mRootNode);

		//glBegin(GL_QUADS);
		//glVertex3f(-10.0f, -10.0f, 10.0f);
		//glVertex3f(-10.0f, 10.0f, 10.0f);
		//glVertex3f(10.0f, 10.0f, 10.0f);
		//glVertex3f(10.0f, -10.0f, 10.0f);
		//glEnd();

		//glBegin(GL_QUADS);
		//glVertex3f(-10.0f, -10.0f, -10.0f);
		//glVertex3f(-10.0f, 10.0f, -10.0f);
		//glVertex3f(10.0f, 10.0f, -10.0f);
		//glVertex3f(10.0f, -10.0f, -10.0f);
		//glEnd();

		//glBegin(GL_QUADS);
		//glVertex3f(-10.0f, -10.0f, 10.0f);
		//glVertex3f(-10.0f, -10.0f, -10.0f);
		//glVertex3f(10.0f, -10.0f, -10.0f);
		//glVertex3f(10.0f, -10.0f, 10.0f);
		//glEnd();

		//glBegin(GL_QUADS);
		//glVertex3f(-10.0f, 10.0f, 10.0f);
		//glVertex3f(-10.0f, 10.0f, -10.0f);
		//glVertex3f(10.0f, 10.0f, -10.0f);
		//glVertex3f(10.0f, 10.0f, 10.0f);
		//glEnd();

		//glBegin(GL_QUADS);
		//glVertex3f(-10.0f, -10.0f, -10.0f);
		//glVertex3f(-10.0f, -10.0f, 10.0f);
		//glVertex3f(-10.0f, 10.0f, 10.0f);
		//glVertex3f(-10.0f, 10.0f, -10.0f);
		//glEnd();

		//glBegin(GL_QUADS);
		//glVertex3f(10.0f, -10.0f, -10.0f);
		//glVertex3f(10.0f, -10.0f, 10.0f);
		//glVertex3f(10.0f, 10.0f, 10.0f);
		//glVertex3f(10.0f, 10.0f, -10.0f);
		//glEnd();
		glEndList();
	}

	glCallList(scene_list);

	glutSwapBuffers();

	do_motion();
}



// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// initial camera
	//if (!img.isOpened())
	//{
	//	std::cout<<"Can not open the camera!"<<std::endl;
	//}
	//cv::namedWindow("Camera",1);
	//img >> image;
	//cv::imshow("Camera", image);

	glutInitWindowSize(900, 600);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInit(&argc, argv);

	glutCreateWindow("Assimp - Very simple OpenGL sample");
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	if (!ImportMs3dFile(modelpath)) return -1;

	//if (!LoadGLTextures(scene))
	//{
	//	std::cout << "Couldn't Load Textures" << std::endl;
	//	return -1;
	//}

	//Initial GL
	//glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glEnable(GL_DEPTH_TEST);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);

	//// XXX docs say all polygons are emitted CCW, but tests show that some aren't.
	//if (getenv("MODEL_IS_BROKEN"))
	//	glFrontFace(GL_CW);

	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	glutGet(GLUT_ELAPSED_TIME);
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