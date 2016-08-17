/*
	2015/02/09:如果需要重复使用GPU显存，考虑减少GPU的内存分配和GPU与CPU之间的数据传递(未解决)
*/

//#include "CNormalLens.h"
//#include "CFisheyeLens.h"
#include "CSurroundView.h"
#include <iostream>

using namespace std;

int main()
{
	//CNormalLens test;
	//Point pt, pt_dst;
	//pt.x = 1;
	//pt.y = 1;
	//pt_dst = test.PerspectivePoint(pt,0,0,0,0,0,1);
	//cout<<pt_dst.x<<"\n";
	//cout<<pt_dst.y<<"\n";

	//Mat src = imread("test.bmp",1);
	//Mat dst;
	//for (int theta = -9; theta <= 9; theta++)
	//{
	//	test.PerspectiveImage(src, dst, theta*10, 0, 0, 0, 0, 2);
	//	imshow("src", src);
	//	waitKey(300);
	//	imshow("dst", dst);
	//	waitKey(300);
	//}

	//CFisheyeLens test1;
	Mat src = imread("front45.bmp");
	//test1.Initial("calib_results_f.txt");
	//
	//Size src_size = Size((int)src.cols, (int)src.rows);

	//Mat mapx = Mat(src_size, CV_32FC1);
	//Mat mapy = Mat(src_size, CV_32FC1);

	//Mat dst = src;

	//test1.Create_Undistort_LUT(mapx,mapy,0,500,4,1);

	//remap(src, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
	//imshow("dst", dst);
	//waitKey();

	//test1.Create_UndistortPerspective_LUT(mapx,mapy,30,0,0,0,0,2,1);
	//remap(src, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, 0);
	//imshow("dst", dst);
	//waitKey();

	CSurroundView test2;

	//test2.Get_LUT("map_front.bmp","map_frontleft.bmp","map_frontright.bmp","map_rear.bmp","map_rearleft.bmp","map_rearright.bmp");
	test2.Get_LUT("map_front.tif","map_frontleft.tif","map_frontright.tif","map_rear.tif","map_rearleft.tif","map_rearright.tif");


	//test2.Get_ocammodel("calib_results_f.txt","calib_results_lf_new.txt","calib_results_rf.txt","calib_results_r.txt","calib_results_lb_new.txt","calib_results_rb.txt");
	//test2.Get_persparas("Parameters/pre_paras.txt");
	//test2.Get_blendcoeff("Parameters/blendcoeff_front.txt","Parameters/blendcoeff_frontleft.txt","Parameters/blendcoeff_frontright.txt",
	//		"Parameters/blendcoeff_rear.txt","Parameters/blendcoeff_rearleft.txt","Parameters/blendcoeff_rearright.txt");
	//test2.Create_6_UndistortPerspective_LUT();
	Mat dst = Mat(Size(2304, 2944), src.type(), Scalar(0,0,0));
	imshow("src", src);
	waitKey();
	test2.Cal_SurroundView(src, imread("frontleft45.bmp"), imread("frontright45.bmp"), imread("rear45.bmp"), imread("rearleft45.bmp"), imread("rearright45.bmp"), dst);
	imshow("dst", dst);
	imwrite("dst.bmp", dst);
	waitKey();
	system("pause");
	return 0;
}