#pragma once
#ifndef GREY_ASSITED_H
#define GREY_ASSITED_H
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//using namespace cv;

void decode_gp_single(bool** D, cv::Mat& wrappedPhaseMap, double** Pro_point, int image_count, char* path/*, double** phase1*/);

#endif