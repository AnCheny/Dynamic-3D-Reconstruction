#pragma once

// #include "FlyCapture2.h"
#include "math.h"
#include "string.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <fstream>
#include <iomanip>

// using namespace FlyCapture2;
//using namespace cv;
using namespace std;


//Point2d(x,y) == y represents row, x represents col
//Size(width, height)

// TODO£ºdefine global variable
#define PI 3.1415926
struct decode_pattern{
	int phase_num;
	int cycle;
	double threshold;
	double phase_offset;
	int board_row;
	int board_col;
	double d_y;
	double d_x;
};

const int cam_width = 1280, cam_height = 1024;
//const int cam_width = 720, cam_height = 540;
//const int cam_width = 912, cam_height = 1140;
const int pro_width = 912, pro_height = 1140;

const decode_pattern dec_square = {6, 20, 0.5, -0.715741, 9, 13, 40, 40};// Square ChessBoard

const decode_pattern dec_circle = {6, 20, 0.5, -0.715741, 9, 11, 50, 50};// Circle ChessBoard

const decode_pattern dec_config = {3, 20, 0.5, 0.7, 0, 0, 0, 0 };// 4,0.5

const decode_pattern dec_config_2 = { 3, 20, 0.5, 0, 0, 0, 0, 0 };

const int cal_image_serial = ceil(log2(pro_width / dec_circle.cycle)) + ceil(log2(pro_height / dec_circle.cycle)) + 2; // the serial number of image which is used to extract corners
const int single_image_serial = ceil(log2(pro_width / dec_config.cycle)) + 2;
const int double_image_serial = ceil(log2(pro_width / dec_config.cycle)) + ceil(log2(pro_height / dec_config.cycle)) + 2;

// reconstruct using horizonal and vertical stripes
int decoding_3D_double();
void decode_gp_double(bool** D, cv::Point2f** Pro_point, int image_count, char* path);

// reconstruct using only horizonal stripes
int decoding_3D_single();
void decode_gp_single(bool** D, double ** Pro_point, int image_count, char* path);


// for test calibration
void decode_gp_double_calib(bool** D, cv::Point2f** Pro_point, int image_count, char* path);

// int get_cam_image();		   //capture image of Commercial Projector 

int cap_calib_image();	       //capture image of DLP4500: DLP4500 trigger out/ Camera triggger in to calibrate system

int cap_restruct_image();	   // capture image to restruct the model

int activate_restruct_image();  // Camera Trigger out / DLP4500 Trigger in


void cam_pro_calib_square();   //square chessboard is used to calibrate system

void cam_pro_calib_circle();   //circle chessboard is used to calibrate system

void cam_pro_calib_matlab();   //point1.txt/point2.txt is used to calibarate system



////////////////////////////////
//// for testing circle detector of projector  /////
vector<cv::Point2f> decode_corner_bilinear(vector<cv::Point2f>cam_point, int image_count, int point_num);
void compute_save_result(vector<vector<cv::Point3f>> object_points, vector<vector<cv::Point2f>> cam_points_seq,
	cv::Mat cameraMatrix1, cv::Mat distCoeffs1, vector<cv::Mat>rvecsMat, vector<cv::Mat>tvecsMat, string file_name, int image_num, cv::Size board_size);
void circle_map_calib();
void circle_blob_test();
void search_app_th();
////////////////////////////////

void detect_corner();	//detect corners of chessboard
vector<float> detect_sub(vector<cv::Point2f>cam_point, int image_count, int point_num);
vector<cv::Point2f> detect_sub_2(vector<cv::Point2f>cam_point, int image_count, int point_num);
