/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "..\\include\\opencv2\\phase_unwrapping\\histogramphaseunwrapping.hpp"
#include "..\\include\\opencv2\\structured_light_top.hpp"
#include "..\\include\\opencv2\\para_config.h"
#include "..\\include\\opencv2\\grey_assited.h"
#include"..\\include\\opencv2\\dual_frequency.h"
//#include "FTP.h"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//#include <pcl/io/pcd_io.h>


//
//using namespace cv;
using namespace std;
using namespace cv::phase_unwrapping;
static const char* keys =
{
    "{@width | | Projector width}"
    "{@height | | Projector height}"
    "{@periods | | Number of periods}"
    "{@setMarkers | | Patterns with or without markers}"
    "{@horizontal | | Patterns are horizontal}"
    "{@methodId | | Method to be used}"
    "{@outputPatternPath | | Path to save patterns}"
    "{@outputWrappedPhasePath | | Path to save wrapped phase map}"
    "{@outputUnwrappedPhasePath | | Path to save unwrapped phase map}"
    "{@outputCapturePath | | Path to save the captures}"
    "{@reliabilitiesPath | | Path to save reliabilities}"
};
static void help()
{
    cout << "\nThis example generates sinusoidal patterns" << endl;
    cout << "To call: ./example_structured_light_createsinuspattern <width> <height>"
        " <number_of_period> <set_marker>(bool) <horizontal_patterns>(bool) <method_id>"
        " <output_captures_path> <output_pattern_path>(optional) <output_wrapped_phase_path> (optional)"
        " <output_unwrapped_phase_path>" << endl;
}
extern "C" int pcd_show();
int main(int argc, char** argv)
{
    cv::structured_light::SinusoidalPattern::Params params;
    cv::phase_unwrapping::HistogramPhaseUnwrapping::Params paramsUnwrapping;
    cv::String outputCapturePath;
    cv::String outputPatternPath;
    cv::String outputWrappedPhasePath;
    cv::String outputUnwrappedPhasePath;
    cv::String reliabilitiesPath;
    if (argc < 2)
    {
        //params.width = 720;//912 1280
        params.width = 1280;//912 1280
        //params.height = 540;//1140 1024
        params.height = 1024;//1140 1024
        //params.width = 1280;//912 1280
        //params.height = 1024;//1140 1024
        params.nbrOfPeriods = 20;//46
        params.setMarkers = 0;//0
        params.horizontal = 0;//0 = vertival
        params.methodId = 3;//3 = FAPS_copy
        params.shiftValue = static_cast<float>(2*CV_PI / 3);
        params.nbrOfPixelsBetweenMarkers = 70;
        outputCapturePath = "E:\\vs_engineering\\structured_light\\structured_light\\out\\outputCapturePath\\";
        outputPatternPath = "E:\\vs_engineering\\structured_light\\structured_light\\out\\outputPatternPath\\";
        outputWrappedPhasePath = "E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\out\\OutputWrappedPhase\\";
        outputUnwrappedPhasePath = "E:\\vs_engineering\\structured_light\\structured_light\\out\\OutputUnwrappedPhase\\";
        reliabilitiesPath = "E:\\vs_engineering\\structured_light\\structured_light\\out\\ReliabilitiesPath\\";
    }
    else {
        // Retrieve parameters written in the command line
        cv::CommandLineParser parser(argc, argv, keys);
        params.width = parser.get<int>(0);
        params.height = parser.get<int>(1);
        params.nbrOfPeriods = parser.get<int>(2);
        params.setMarkers = parser.get<bool>(3);
        params.horizontal = parser.get<bool>(4);
        params.methodId = parser.get<int>(5);
        outputCapturePath = parser.get<cv::String>(6);
        //params.shiftValue = static_cast<float>(2 * CV_PI / 3);
        params.shiftValue = static_cast<float>(2*CV_PI / 3);
        params.nbrOfPixelsBetweenMarkers = 70;
        outputPatternPath = parser.get<cv::String>(7);
        outputWrappedPhasePath = parser.get<cv::String>(8);
        outputUnwrappedPhasePath = parser.get<cv::String>(9);
        reliabilitiesPath = parser.get<cv::String>(10);
    }

    cv::Ptr<cv::structured_light::SinusoidalPattern> sinus =
        cv::structured_light::SinusoidalPattern::create(cv::makePtr<cv::structured_light::SinusoidalPattern::Params>(params));
    

    cv::Ptr<cv::phase_unwrapping::HistogramPhaseUnwrapping> phaseUnwrapping;
    vector<cv::Mat> patterns;//
    cv::Mat shadowMask;
    cv::Mat unwrappedPhaseMap, unwrappedPhaseMap8;// trans to 8 bit

    //Mat wrappedPhaseMap = Mat::zeros(cam_height, cam_width, CV_32FC1);

    cv::Mat wrappedPhaseMap, wrappedPhaseMap8, wrappedPhaseMap_low, PSPwrappedPhaseMap, PSPwrappedPhaseMap_low;
    //Generate sinusoidal patterns
    sinus->generate(patterns);// output the generated images, such the sin;



    //VideoCapture cap(CAP_PVAPI);// this is the device 
    //VideoCapture cap(CAP_ANY);// this is  CAP_ANY = auto
    //if (!cap.isOpened())
    //{
    //    cout << "Camera could not be opened" << endl;
    //    return -1;
    //}
    //cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_MONO8);// 306 1 set the camera 
    //namedWindow("pattern", WINDOW_NORMAL);
    //setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    //imshow("pattern", patterns[0]);
    //cout << "Press any key when ready" << endl;
    //waitKey(0);
    int nbrOfImages = 3;
    int count = 0;
    vector<cv::Mat> img(nbrOfImages);
    cv::Size camSize(-1, -1);
    //while (count < nbrOfImages)
    //{
    //    for (int i = 0; i < (int)patterns.size(); ++i)
    //    {
    //        imshow("pattern", patterns[i]);
    //        waitKey(300);//wait 300  ms
    //        cap >> img[count];//2维矩阵
    //        count += 1;
    //    }
    //}
    //cout << "press enter when ready" << endl;
    bool loop = true;
    //while (loop)
    //{
    //    char c = (char)waitKey(0);
    //    if (c == 10)
    //    {
    //        loop = false;
    //        destroyWindow("pattern");
    //    }
    //}
    cout << "Process begin..." << endl;
    switch (params.methodId)
    {
    case cv::structured_light::FTP:
        for (int i = 0; i < nbrOfImages; ++i)
        for (int i = 0; i < 1; ++i)
        {
            /*We need three images to compute the shadow mask, as described in the reference paper
             * even if the phase map is computed from one pattern only
            */
            vector<cv::Mat> captures;// used to save the 
            if (i == nbrOfImages - 2)
            {
                captures.push_back(img[i]);
                captures.push_back(img[i - 1]);
                captures.push_back(img[i + 1]);
            }
            else if (i == nbrOfImages - 1)
            {
                captures.push_back(img[i]);
                captures.push_back(img[i - 1]);
                captures.push_back(img[i - 2]);
            }
            else
            {
                captures.push_back(img[i]);
                captures.push_back(img[i + 1]);
                captures.push_back(img[i + 2]);
            }
            //sinus->computePhaseMap(captures, wrappedPhaseMap, shadowMask);
            if (camSize.height == -1)
            {
                camSize.height = img[i].rows;
                camSize.width = img[i].cols;
                paramsUnwrapping.height = camSize.height;
                paramsUnwrapping.width = camSize.width;
                phaseUnwrapping =
                    cv::phase_unwrapping::HistogramPhaseUnwrapping::create(paramsUnwrapping);
            }
            sinus->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);
            phaseUnwrapping->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, shadowMask);
            cv::Mat reliabilities, reliabilities8;
            phaseUnwrapping->getInverseReliabilityMap(reliabilities);
            reliabilities.convertTo(reliabilities8, CV_8U, 255, 128);
            ostringstream tt;
            tt << i;
            imwrite(reliabilitiesPath + tt.str() + ".png", reliabilities8);
            unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);
            wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);
            if (!outputUnwrappedPhasePath.empty())    while (loop)
            {
                char c = (char)cv::waitKey(0);
                if (c == 10)
                {
                    loop = false;
                    cv::destroyWindow("pattern");
                }
            }
            {
                ostringstream name;
                name << i;
                imwrite(outputUnwrappedPhasePath + "_FTP_" + name.str() + ".png", unwrappedPhaseMap8);
            }
            if (!outputWrappedPhasePath.empty())
            {
                ostringstream name;
                name << i;
                imwrite(outputWrappedPhasePath + "_FTP_" + name.str() + ".png", wrappedPhaseMap8);
            }
        }
        break;
    case cv::structured_light::PSP:
    case cv::structured_light::FAPS:
        //for (int i = 0; i < nbrOfImages - 2; ++i)
    {
        cv::Mat pic1, pic2, pic3;
        pic1 = cv::imread("C:\\Users\\DELL\\OneDrive\\桌面\\zby_code\\decode_image\\1_2.bmp");
        pic2 = cv::imread("C:\\Users\\DELL\\OneDrive\\桌面\\zby_code\\decode_image\\1_4.bmp");
        pic3 = cv::imread("C:\\Users\\DELL\\OneDrive\\桌面\\zby_code\\decode_image\\1_6.bmp");
        //imshow("pic1", pic1);
        cvtColor(pic1, pic1, CV_BGR2GRAY);
        cvtColor(pic2, pic2, CV_BGR2GRAY);
        cvtColor(pic3, pic3, CV_BGR2GRAY);
        vector<cv::Mat> captures;
        vector<cv::Mat> theta;
        vector<cv::Mat> temps;

        //captures.push_back(img[i]);
        //captures.push_back(img[i + 1]);
        //captures.push_back(img[i + 2]);
        // 1.设置Mat变量
        // 2.读入图片
        // 3.输入到capture
        captures.push_back(pic1);
        captures.push_back(pic2);
        captures.push_back(pic3);
        //sinus->computePhaseMap(captures, wrappedPhaseMap, shadowMask); 
        cv::FileStorage fs("C:/Users/DELL/OneDrive/桌面/zby_code/output/outputs/parameter.xml", cv::FileStorage::READ);
        cv::FileStorage fs_p("C:/Users/DELL/OneDrive/桌面/zby_code/output/outputs/undistort_point.xml", cv::FileStorage::READ);
        
        bool** D = new bool* [cam_height];
        double** Pro_point = new double* [cam_height];
        for (int i = 0; i < cam_height; i++)
        {
            D[i] = new bool[cam_width];
            Pro_point[i] = new double[cam_width];//height个指针，指向weidth个指针。内存着投影仪的点坐标
        }

        char path[100] = "C:\\Users\\DELL\\OneDrive\\桌面\\zby_code\\decodeimage\\";
        int image_count = 1;

        //decode_gp_single(D, wrappedPhaseMap, Pro_point, image_count, path/*, double** phase1*/);


        // revised by yj
        int m, n;
        int count = 0;
        int k1, k2;
        float med_x;
        cv::Mat win_x = cv::Mat::zeros(cv::Size(25, 1), CV_32FC1);
        cv::Mat win_xx = cv::Mat::zeros(cv::Size(25, 1), CV_32FC1);

        int k, l;
        cv::Mat /*R[3][3], T[3] = { 0, 0, 0 }*/ T1;

        /*distort the undistort_cam*/
        cv::Mat undistort_cam;
        cv::Mat cameraMatrix1 = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)); /* 摄像机内参数矩阵 */
        cv::Mat distCoeffs1 = cv::Mat(1, 5, CV_64FC1, cv::Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2 */
        cv::Mat cameraMatrix2 = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)); /* 投影仪内参数矩阵 */
        cv::Mat distCoeffs2 = cv::Mat(1, 5, CV_64FC1, cv::Scalar::all(0)); /* 投影仪的5个畸变系数：k1,k2,p1,p2 */
        cv::Mat R, T;
        cv::Mat A = cv::Mat(3, 3, CV_64FC1);
        cv::Mat B = cv::Mat(3, 1, CV_64FC1);
        cv::Mat RT1 = cv::Mat(3, 4, CV_64FC1);
        cv::Mat RT2 = cv::Mat(3, 4, CV_64FC1);

        if (fs.isOpened())
        {
            fs["cameraMatrix1"] >> cameraMatrix1;
            distCoeffs1 = fs["distCoeffs1"].mat();
            cameraMatrix2 = fs["cameraMatrix2"].mat();
            distCoeffs2 = fs["distCoeffs2"].mat();
            R = fs["R"].mat();
            T = fs["T"].mat();
        }
        else
        {
            cout << "Can't open parameter.xml" << endl;
        }

        if (fs_p.isOpened())
        {
            fs_p["undistort_cam"] >> undistort_cam;
        }
        else
        {
            cout << "Can't open parameter.xml" << endl;
        }

        char point_path[100];
        sprintf(point_path, "./points_single%d.txt", 1);
        ofstream fout(point_path);

        fs.release(); fs_p.release();
        cv::Mat temp4 = cv::Mat::zeros(3, 1, CV_64FC1);
        hconcat(cameraMatrix1, temp4, RT1);
        cv::Mat temp5;
        vconcat(R.t(), T.t(), temp5);
        RT2 = cameraMatrix2 * temp5.t();
        /*distort the undistort_cam*/


        /*restruct the 3D poinnt*/

        char path1[100];
        char sub_name[20];
        strcpy(path1, path);
        sprintf(sub_name, "%d_%d.bmp", 1, single_image_serial);
        strcat(path1, sub_name);
        cv::Mat image = cv::imread(path1, cv::IMREAD_GRAYSCALE);
        image.convertTo(image, CV_64FC1);

        for (int i = 50; i < cam_height - 50; i++)
        {
            for (int j = 50; j < cam_width - 50; j++)
            {
                if (D[i][j] == 1)
                {
                    cv::Point2d point1 = undistort_cam.at<cv::Vec2d>(i, j);

                    double cam_i = point1.y;
                    double cam_j = point1.x;
                    double proj_j = Pro_point[i][j];
                    //double proj_j = point1.x;

                    A.at<double>(0, 0) = RT1.at<double>(0, 0) - RT1.at<double>(2, 0) * cam_j;
                    A.at<double>(0, 1) = RT1.at<double>(0, 1) - RT1.at<double>(2, 1) * cam_j;
                    A.at<double>(0, 2) = RT1.at<double>(0, 2) - RT1.at<double>(2, 2) * cam_j;

                    A.at<double>(1, 0) = RT1.at<double>(1, 0) - RT1.at<double>(2, 0) * cam_i;
                    A.at<double>(1, 1) = RT1.at<double>(1, 1) - RT1.at<double>(2, 1) * cam_i;
                    A.at<double>(1, 2) = RT1.at<double>(1, 2) - RT1.at<double>(2, 2) * cam_i;

                    A.at<double>(2, 0) = RT2.at<double>(0, 0) - RT2.at<double>(2, 0) * proj_j;
                    A.at<double>(2, 1) = RT2.at<double>(0, 1) - RT2.at<double>(2, 1) * proj_j;
                    A.at<double>(2, 2) = RT2.at<double>(0, 2) - RT2.at<double>(2, 2) * proj_j;

                    B.at<double>(0, 0) = RT1.at<double>(2, 3) * cam_j - RT1.at<double>(0, 3);
                    B.at<double>(1, 0) = RT1.at<double>(2, 3) * cam_i - RT1.at<double>(1, 3);
                    B.at<double>(2, 0) = RT2.at<double>(2, 3) * proj_j - RT2.at<double>(0, 3);

                    cv::Mat d_point = (A.inv()) * B;


                    fout << d_point.at<double>(0, 0) << ' ' << d_point.at<double>(1, 0)
                        << ' ' << d_point.at<double>(2, 0) << ' ' << image.at<double>(i, j) << endl;
                 }
             }
        }
            cout << "Frame " << 1 << ": finished" << endl;
            fout.close();

           // if (camSize.height == -1)
           // {
           //     //camSize.height = img[i].rows;
           //     //camSize.width = img[i].cols;
           //     camSize.height = pic1.rows;
           //     camSize.width = pic1.cols;
           //     paramsUnwrapping.height = camSize.height;
           //     paramsUnwrapping.width = camSize.width;
           //     phaseUnwrapping =
           //         phase_unwrapping::HistogramPhaseUnwrapping::create(paramsUnwrapping);
           // }

           // sinus->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);//first use the unwrapped
           // unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);
           // wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);
           // phaseUnwrapping->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, shadowMask);//second use the unwrapped
           // Mat reliabilities, reliabilities8;//create the reliable img
           // phaseUnwrapping->getInverseReliabilityMap(reliabilities);
           // reliabilities.convertTo(reliabilities8, CV_8U, 255, 128);
           /////////***********//
           // //unwrappedPhaseMap8.3Dpointcloud()
           // ostringstream tt;
           // tt << 0;
           // //tt << i;
           // imwrite(reliabilitiesPath + tt.str() + ".png", reliabilities8);

           // // trans the phasemap to the point img
           // // 
           // //std::vector<Pixel> pixs;
           // //cv::phase_unwrapping::HistogramPhaseUnwrapping_Impl::Pixel() point_img;
           // //point_img -> 

           // /////////

           // if (!outputUnwrappedPhasePath.empty())
           // {
           //     ostringstream name;
           //     name << 0;
           //     //name << i;
           //     if (params.methodId == structured_light::PSP)
           //         imwrite(outputUnwrappedPhasePath + "_PSP_" + name.str() + ".png", unwrappedPhaseMap8);
           //     else
           //         imwrite(outputUnwrappedPhasePath + "_FAPS_" + name.str() + ".png", unwrappedPhaseMap8);
           // }
           // if (!outputWrappedPhasePath.empty())
           // {
           //     ostringstream name;
           //     name << 0;
           //     //name << i;
           //     if (params.methodId == structured_light::PSP)
           //         imwrite(outputWrappedPhasePath + "_PSP_" + name.str() + ".png", wrappedPhaseMap8);
           //     else
           //         imwrite(outputWrappedPhasePath + "_FAPS_" + name.str() + ".png", wrappedPhaseMap8);
           // }
           // if (!outputCapturePath.empty())
           // {
           //     ostringstream name;
           //     name << 0;
           //     //name << i;
           //     //if (params.methodId == structured_light::PSP)
           //     //    imwrite(outputCapturePath + "_PSP_" + name.str() + ".png", img[i]);
           //     //else
           //     //    imwrite(outputCapturePath + "_FAPS_" + name.str() + ".png", img[i]);
           //     //if (i == nbrOfImages - 3)
           //     //{
           //     //    if (params.methodId == structured_light::PSP)
           //     //    {
           //     //        ostringstream nameBis;
           //     //        nameBis << i + 1;
           //     //        ostringstream nameTer;
           //     //        nameTer << i + 2;
           //     //        //imwrite(outputCapturePath + "_PSP_" + nameBis.str() + ".png", img[i + 1]);
           //     //        //imwrite(outputCapturePath + "_PSP_" + nameTer.str() + ".png", img[i + 2]);
           //     //    }
           //     //    else
           //     //    {
           //     //        ostringstream nameBis;
           //     //        nameBis << i + 1;
           //     //        ostringstream nameTer;
           //     //        nameTer << i + 2;
           //     //        //imwrite(outputCapturePath + "_FAPS_" + nameBis.str() + ".png", img[i + 1]);
           //     //        //imwrite(outputCapturePath + "_FAPS_" + nameTer.str() + ".png", img[i + 2]);
           //     //    }
           //     //}
           // }
        }
        break;
    case cv::structured_light::FAPS_copy:
    {
        cv::Mat pic1, pic2, pic3, pic4, pic5,pic6;
        pic1 = cv::imread("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-12-05\\pics\\static\\0001.bmp");
        pic2 = cv::imread("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-12-05\\pics\\static\\0002.bmp");
        pic3 = cv::imread("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-12-05\\pics\\static\\0003.bmp");
        pic4 = cv::imread("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-12-05\\pics\\static\\0004.bmp");
        pic5 = cv::imread("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-12-05\\pics\\static\\0005.bmp");
        pic6 = cv::imread("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-12-05\\pics\\static\\0006.bmp");
        //imshow("pic1", pic1);
        cvtColor(pic1, pic1, CV_BGR2GRAY);
        cvtColor(pic2, pic2, CV_BGR2GRAY);
        cvtColor(pic3, pic3, CV_BGR2GRAY);
        cvtColor(pic4, pic4, CV_BGR2GRAY);
        cvtColor(pic5, pic5, CV_BGR2GRAY);
        cvtColor(pic6, pic6, CV_BGR2GRAY);
        vector<cv::Mat> captures, captures1;
        //captures.push_back(pic4);
        //captures.push_back(pic5);
        //captures.push_back(pic6);
        //captures1.push_back(pic1);
        //captures1.push_back(pic2);
        //captures1.push_back(pic3);
        captures1.push_back(pic4);
        captures1.push_back(pic5);
        captures1.push_back(pic6);
        captures.push_back(pic1);
        captures.push_back(pic2);
        captures.push_back(pic3);
        //Mat** wrappedPhaseMap = new double* [params.height];
        //double** wrappedPhaseMap_low = new double* [params.height];
        //for (int i = 0; i < params.height; i++)
        //{
        //    wrappedPhaseMap[i] = new double[params.width];
        //    wrappedPhaseMap_low[i] = new double[params.width];//height个指针，指向weidth个指针。内存着投影仪的点坐标
        //}

        sinus->computePhaseMap(captures,captures1, wrappedPhaseMap, wrappedPhaseMap_low, PSPwrappedPhaseMap, PSPwrappedPhaseMap_low,shadowMask);//modified
        cv::Mat shadowMask_8, wrappedPhaseMap_low8;
        shadowMask.convertTo(shadowMask_8, CV_8U, 255, 128);
        imwrite("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\out\\OutputWrappedPhase\\shadowMask.png", shadowMask_8);
        wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);
        imwrite(outputWrappedPhasePath + "high_frequency.png", wrappedPhaseMap8);
        wrappedPhaseMap_low.convertTo(wrappedPhaseMap_low8, CV_8U, 255, 128);
        imwrite(outputWrappedPhasePath + "low_frequency.png", wrappedPhaseMap_low8);
        cout << "the wrapped phase is out!" << endl;

        //// output the gray of the pic
        //ofstream fout3("./matlab/wrappedPhasemap_low_480.txt");
        //ofstream fout1("./matlab/wrappedPhasemap_high_480.txt");
        //for (int i = 0; i < cam_width; i++)
        //{
        //    fout3 << wrappedPhaseMap_low8.at<float>(380, i) << endl;
        //    fout1 << wrappedPhaseMap8.at<float>(380, i) << endl;
        //}
        //fout3.close();
        //fout1.close();
        ////
        cv::Mat img = cv::imread("..\\out\\OutputWrappedPhase\\high_frequency.png", 0);
        ofstream fout2(".\\matlab\\wrapped_line_480.txt");
                for (int i = 0; i < cam_width; i++)
        {
            fout2 << wrappedPhaseMap_low.at<float>(380, i) << endl;
            //fout2 << (int)img.at<unsigned char>(380,i) << endl;
            //cout << (int)img.at<unsigned char>(380,i) << " " << endl;
        }
        fout2.close();
        //cout << "the wrappedphase has saved !" << endl;
        /*if (camSize.height == -1)
        {
            camSize.height = img[i].rows;
            camSize.width = img[i].cols;
            paramsUnwrapping.height = camSize.height;
            paramsUnwrapping.width = camSize.width;
            phaseUnwrapping =
                cv::phase_unwrapping::HistogramPhaseUnwrapping::create(paramsUnwrapping);
        }*/

        cv::Mat undistort_cam;
        cv::Mat cameraMatrix1 = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)); /* 摄像机内参数矩阵 */
        cv::Mat distCoeffs1 = cv::Mat(1, 5, CV_64FC1, cv::Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2 */
        cv::Mat cameraMatrix2 = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0)); /* 投影仪内参数矩阵 */
        cv::Mat distCoeffs2 = cv::Mat(1, 5, CV_64FC1, cv::Scalar::all(0)); /* 投影仪的5个畸变系数：k1,k2,p1,p2 */
        cv::FileStorage fs("E:/vs_engineering/structured_light/strctured_light_revised/structured_light/calibration/parameter.xml", cv::FileStorage::READ);
        cv::FileStorage fs_p("E:/vs_engineering/structured_light/strctured_light_revised/structured_light/calibration/undistort_point.xml", cv::FileStorage::READ);
        /*cv::FileStorage fs("C:/Users/DELL/OneDrive/桌面/zby_code/output/outputs/parameter.xml", cv::FileStorage::READ);
        cv::FileStorage fs_p("C:/Users/DELL/OneDrive/桌面/zby_code/output/outputs/undistort_point.xml", cv::FileStorage::READ);*/
        cv::Mat R, T;
        cv::Mat A = cv::Mat(3, 3, CV_64FC1);
        cv::Mat B = cv::Mat(3, 1, CV_64FC1);
        cv::Mat RT1 = cv::Mat(3, 4, CV_64FC1);
        cv::Mat RT2 = cv::Mat(3, 4, CV_64FC1);

        if (fs.isOpened())
        {
            fs["cameraMatrix1"] >> cameraMatrix1;
            distCoeffs1 = fs["distCoeffs1"].mat();
            cameraMatrix2 = fs["cameraMatrix2"].mat();
            distCoeffs2 = fs["distCoeffs2"].mat();
            R = fs["R"].mat();
            T = fs["T"].mat();
        }
        else
        {
            cout << "Can't open parameter.xml" << endl;
        }

        if (fs_p.isOpened())
        {
            fs_p["undistort_cam"] >> undistort_cam;
        }
        else
        {
            cout << "Can't open parameter.xml" << endl;
        }

        char point_path[100];
        sprintf(point_path, "./points_single_2%d.txt", 1);
        ofstream fout(point_path);

        fs.release(); fs_p.release();
        cv::Mat temp4 = cv::Mat::zeros(3, 1, CV_64FC1);
        hconcat(cameraMatrix1, temp4, RT1);
        cv::Mat temp5;
        vconcat(R.t(), T.t(), temp5);
        RT2 = cameraMatrix2 * temp5.t();
        /*distort the undistort_cam*/


        /*restruct the 3D poinnt*/
        //char path[100] = "C:\\Users\\DELL\\OneDrive\\桌面\\zby_code\\decodeimage\\";
        char path[200] = "E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\documents-export-2022-11-14\\13_3000\\";
        //bool** D = new bool* [params.height];
        double** Pro_point = new double* [params.height];
        for (int i = 0; i < params.height; i++)
        {
            //D[i] = new bool[params.width];
            Pro_point[i] = new double[params.width];//height个指针，指向weidth个指针。内存着投影仪的点坐标
        }

        dual_frequency(shadowMask_8,wrappedPhaseMap, wrappedPhaseMap_low, PSPwrappedPhaseMap, PSPwrappedPhaseMap_low,Pro_point, 1, path/*, double** phase1*/);
        //decode_gp_single(D, wrappedPhaseMap, Pro_point, 1, path/*,phase1*/);

        pcl::PointCloud<pcl::PointXYZ> cloud2;
        //将上面的数据填入点云
        cloud2.width = params.width;//设置点云宽度
        cloud2.height = params.height; //设置点云高度
        cloud2.is_dense = false; //非密集型
        cloud2.points.resize(long(cloud2.width * cloud2.height)); //变形，无序

        char path1[200];
        char sub_name[20];

        //strcpy(path1, path);
        ////sprintf(sub_name, "0006.bmp", 1, single_image_serial);
        //sprintf(sub_name, "0005.bmp");
        //strcat(path1, sub_name);
        //cv::Mat image = cv::imread(path1, cv::IMREAD_GRAYSCALE);
        //image.convertTo(image, CV_64FC1);
        int pointsize = 0;
        for (int i3 = 50; i3 <= params.height - 50; i3++)
        {
            for (int j3 = 50; j3 <= params.width - 50; j3++)
            {
                if (shadowMask_8.at<uchar>(i3, j3) > 200)
                //if (D[i3][j3] == 1)
                {

                    cv::Point2d point1 = undistort_cam.at<cv::Vec2d>(i3, j3); //distort the point

                    double cam_i = point1.y;
                    double cam_j = point1.x;
                    double proj_j = Pro_point[i3][j3];

                    A.at<double>(0, 0) = RT1.at<double>(0, 0) - RT1.at<double>(2, 0) * cam_j;
                    A.at<double>(0, 1) = RT1.at<double>(0, 1) - RT1.at<double>(2, 1) * cam_j;
                    A.at<double>(0, 2) = RT1.at<double>(0, 2) - RT1.at<double>(2, 2) * cam_j;
                    A.at<double>(1, 0) = RT1.at<double>(1, 0) - RT1.at<double>(2, 0) * cam_i;
                    A.at<double>(1, 1) = RT1.at<double>(1, 1) - RT1.at<double>(2, 1) * cam_i;
                    A.at<double>(1, 2) = RT1.at<double>(1, 2) - RT1.at<double>(2, 2) * cam_i;
                        
                    A.at<double>(2, 0) = RT2.at<double>(0, 0) - RT2.at<double>(2, 0) * proj_j;
                    A.at<double>(2, 1) = RT2.at<double>(0, 1) - RT2.at<double>(2, 1) * proj_j;
                    A.at<double>(2, 2) = RT2.at<double>(0, 2) - RT2.at<double>(2, 2) * proj_j;
                        
                    B.at<double>(0, 0) = RT1.at<double>(2, 3) * cam_j - RT1.at<double>(0, 3);
                    B.at<double>(1, 0) = RT1.at<double>(2, 3) * cam_i - RT1.at<double>(1, 3);
                    B.at<double>(2, 0) = RT2.at<double>(2, 3) * proj_j - RT2.at<double>(0, 3);

                    //得到给定矩阵src的逆矩阵保存到des中。//按第一行展开计算|A|

                    cv::Mat A_t = A.t();
                    cv::Mat d_point = ((A_t * A).inv()) * A_t * B;
                    //GetMatrixInverse(A, A_1);
                    //Mat d_point = (A_1)*B;



                    /*cloud2.points[pointsize].x = D_1[0][2] * E[2] + D_1[0][1] * E[1] + D_1[0][0] * E[0];
                    cloud2.points[pointsize].y = D_1[1][2] * E[2] + D_1[1][1] * E[1] + D_1[1][0] * E[0];
                    cloud2.points[pointsize].z = D_1[2][2] * E[2] + D_1[2][1] * E[1] + D_1[2][0] * E[0];*/
                    cloud2.points[pointsize].x = d_point.at<double>(0, 0);
                    cloud2.points[pointsize].y = d_point.at<double>(1, 0);
                    cloud2.points[pointsize].z = d_point.at<double>(2, 0);
                    pointsize = pointsize + 1;

                    /*else if (E[2] == 0)
                     {
                         cloud2.points[pointsize].x = D_1[0][1] * E[1];
                         cloud2.points[pointsize].y = D_1[1][1] * E[1];
                         cloud2.points[pointsize].z = D_1[2][1] * E[1];
                         pointsize = pointsize + 1;
                     }*/

                }
            }
        }
        pcl::io::savePCDFileASCII("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\test_pcd_2.pcd", cloud2); //将点云保存到PCD文件中
        std::cout << "pcd has finished and saved!!!! " << endl;
        std::cerr << "Saved " << cloud2.points.size() << " data points to test_pcd.pcd." << std::endl;
    }
            
    break;
    default:
        cout << "error" << endl;
    }
    
    //cout << "done" << endl;
    //if (!outputPatternPath.empty())
    //{
    //    for (int i = 0; i < 3; ++i)
    //    {
    //        ostringstream name;
    //        name << i + 1;
    //        imwrite(outputPatternPath + name.str() + ".png", patterns[i]);
    //    }
    //}
    ////pcd_show();
    ////loop = true;
    ////while (loop)
    ////{
    ////    char key = (char)waitKey(0);
    ////    if (key == 27)
    ////    {
    ////        loop = false;
    ////    }
    ////}

    
    return 0;
}