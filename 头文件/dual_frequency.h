#pragma once
#ifndef DUAL_FREQUENCY_H
#define DUAL_FREQUENCY_H
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//using namespace cv;

void dual_frequency(bool** D, cv::Mat shadowmask,cv::Mat& wrappedPhaseMap, cv::Mat& wrappedPhaseMap_low, cv::Mat PSPwrappedPhaseMap, cv::Mat PSPwrappedPhaseMap_low, double** Pro_point, int image_count, char* path/*, double** phase1*/, cv::Mat shadowMask);

#endif