
#include <cmath>
#include "..\\include\\opencv2\\para_config.h"
#include "..\\include\\opencv2\\grey_assited.h"
#include "..\\include\\opencv2\\phase_unwrapping\\histogramphaseunwrapping.hpp"
#include "..\\include\\opencv2\\structured_light_top.hpp"
#include "..\\include\\opencv2\\para_config.h"
#include "..\\include\\opencv2\\grey_assited.h"
#include"..\\include\\opencv2\\dual_frequency.h"

void dual_frequency(cv::Mat shadowmask, cv::Mat wrappedPhaseMap, cv::Mat wrappedPhaseMap_low,cv::Mat PSPwrappedPhaseMap, cv::Mat PSPwrappedPhaseMap_low, double** Pro_point, int image_count, char* path/*, double** phase1*/)//
{
	double fh = 45.6;//912 / 20 = 57 
	double fl = 44.6;//912 / 44.66 = 20.44843
	double cycleh = 20;
	double cyclel = 20.44843;
	int grey_col = ceil(log2(double(pro_width) / double(dec_config.cycle)));
	int w_i = ceil(double(pro_width) / double(dec_config.cycle));
	int N_w = ceil(log2(w_i));

	double offset_w = floor((pow(2, N_w) - w_i) / 2);

	cv::Mat Max_pixel, Min_pixel;
	//Mat Min_pixel = 255 * Mat::ones(cam_height, cam_width, CV_64FC1);
	//Mat Max_pixel = Mat::zeros(cam_height, cam_width, CV_64FC1);

	char path1[200];
	char path2[200];
	char path3[200];
	char path4[200];
	char sub_name1[20];
	char sub_name2[20];
	char sub_name3[20];
	char sub_name4[20];
	vector<cv::Mat> image_grey(6);
	vector<cv::Mat> image_phase(3);
	//Mat Min_pixel = 255 * Mat::ones(cam_height, cam_width, CV_64FC1);
	//Mat Max_pixel = Mat::zeros(cam_height, cam_width, CV_64FC1);
	//cv::Mat Max_pixel, Min_pixel;

	////* 读取拍摄的图像，存入到image_phase中，用于解码
	//for (int i = 0; i < 11 /*grey_col + dec_config.phase_num + 2*/; i++)	//	gray pattern and white,black 6 +3+2
	//{
	//	int i_x = 2 * i + 1;//1,3,5,7,9,11,13,15,17,19,21,23
	//	int i_y = 2 * (i + 1);//2,4,6,8,10,12,14,16,18,20,22
	//	int temp_phase = i % 3; // 0,1,2
	//	int temp_grey = i % 6;//0,1,2,3,4,5
	//	// grey code 
	//	if (i < 6)
	//	{
	//		sprintf(sub_name1, "000%d.bmp", /*image_count, */(i));
	//		strcpy(path1, path);// path = ./decode_image/
	//		strcat(path1, sub_name1);
	//		cv::Mat temp1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);//读入灰度图
	//		temp1.convertTo(temp1, CV_64FC1);

	//		image_grey.insert(image_grey.begin() + i, temp1);
	//		//image_grey[temp_grey]=temp1.clone();
	//	}

	//	else if (7 < i && i < 11)
	//	{
	//		//phase img
	//		sprintf(sub_name2, "000%d.bmp", /*image_count*/ (i));
	//		strcpy(path2, path);// path = ./decode_image/
	//		strcat(path2, sub_name2);
	//		cv::Mat temp2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);//读入灰度图
	//		temp2.convertTo(temp2, CV_64FC1);

	//		image_phase.insert(image_phase.begin() + (i - 8), temp2);
	//	}
	//}

	//// Min_pixel
	////sprintf(sub_name3, "%d_5.bmp", image_count);
	//sprintf(sub_name3, "0006.bmp");
	//strcpy(path3, path);// path = ./decode_image/
	//strcat(path3, sub_name3);
	//cv::Mat temp3 = cv::imread(path3, cv::IMREAD_GRAYSCALE);//读入灰度图
	//temp3.convertTo(temp3, CV_64FC1);
	//Min_pixel = temp3;

	//// Max_pixel
	////sprintf(sub_name4, "%d_3.bmp", image_count);
	//sprintf(sub_name4, "0007.bmp");
	//strcpy(path4, path);// path = ./decode_image/
	//strcat(path4, sub_name4);
	//cv::Mat temp4 = cv::imread(path4, cv::IMREAD_GRAYSCALE);//读入灰度图
	//temp4.convertTo(temp4, CV_64FC1);
	//Max_pixel = temp4;//灰度值的范围是0-255

	for (int i = 0; i < dec_config.phase_num * 2 ; i++)
	{
		sprintf(sub_name1, "000%d.bmp", /*image_count, */(i+1));//0001~0008
		strcpy(path1, path);// path = ./decode_image/
		strcat(path1, sub_name1);
		cv::Mat temp1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
		temp1.convertTo(temp1, CV_64FC1);
		if (i == 4)
		{
			Min_pixel = temp1;
		}
		if (i == 5)
		{
			Max_pixel = temp1;
		}
		else
		{
			image_phase.push_back(temp1);
		}
	}

	double time0 = static_cast<double>(cv::getTickCount());
	//cv::Mat D_temp = Max_pixel - Min_pixel;
	//for (int i = 0; i < cam_height; i++)
	//{
	//	for (int j = 0; j < cam_width; j++)
	//	{
	//		if ((D_temp.at<double>(i, j)) > 60)
	//			D[i][j] = 1;
	//		else
	//			D[i][j] = 0;
	//	}
	//}

	bool*** dec_img = new bool** [cam_height];
	cv::Point2d** K = new cv::Point2d * [cam_height];
	for (int i = 0; i < cam_height; i++)
	{
		dec_img[i] = new bool* [cam_width];
		K[i] = new cv::Point2d[cam_width];
		for (int j = 0; j < cam_width; j++)
		{
			dec_img[i][j] = new bool[grey_col];
		}
	}

	//D_temp.convertTo(D_temp8, CV_8U, 255, 128);
	//cv::imwrite("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\out\\OutputWrappedPhase\\D_temp.png", D_temp8);

	double** phase1 = new double* [cam_height];
	double** phase12 = new double* [cam_height];
	for (int i = 0; i < cam_height; i++)
	{
		phase1[i] = new double[cam_width];
		phase12[i] = new double[cam_width];
	}
	// k definition
	cv::Point2d** k = new cv::Point2d * [cam_height];
	for (int i = 0; i < cam_height; i++)
	{
		k[i] = new cv::Point2d[cam_width];
	}
	 //k definition
	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 0; j < cam_width; j++)
		{
			if (PSPwrappedPhaseMap.at<float>(i, j) >= PSPwrappedPhaseMap_low.at<float>(i, j))
				phase12[i][j] = PSPwrappedPhaseMap.at<float>(i, j) - PSPwrappedPhaseMap_low.at<float>(i, j);
			else
			{
				phase12[i][j] = 2 * CV_PI-(PSPwrappedPhaseMap_low.at<float>(i, j) - PSPwrappedPhaseMap.at<float>(i, j)) ;
			}
		}
	}

	double k_x[1000];
	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 0; j < cam_width; j++)
		{
			k[i][j].x = ceil(((cycleh /(cyclel - cycleh))*phase12[i][j]- PSPwrappedPhaseMap.at<float>(i,j))/( 2 * CV_PI)+100)-100;
			//k[i][j].x = floor((wrappedPhaseMap_low.at<float>(i, j)*cyclel - wrappedPhaseMap.at<float>(i,j)*cycleh)/ 2 * CV_PI);
			
			}
	}
	const int x = 465;

	ofstream fout7("./matlab/untreated_k_order.txt");
	for (int i = 0; i < cam_width; i++)
	{
		fout7 << k[x][i].x << endl;
	}
	fout7.close();

	//k order
	const int delta = 12;
	for (int i = delta; i < cam_height- delta; i++)
	{
		for (int j = delta; j < cam_width- delta; j++)
		{
			int count = 0;
			/// <summary> the k order 
			for (int k1 = -delta; k1 <= delta; k1++)
			{
				for (int k2 = -delta; k2 <= delta; k2++)
				{
					//if (D[i + k1][j + k2] == 1) {
					k_x[count] = k[i + k1][j + k2].x;
					count = count + 1;
				}
			}
			sort(k_x, k_x + count);
			int med = floor(count / 2);
			if ((k[i][j].x - k_x[med]) > 0.5)
				k[i][j].x -= 1;
			else if ((k[i][j].x - k_x[med]) < -0.5)
				k[i][j].x += 1;
		}
	}

	//修正条纹阶数K
	int temp[cam_height][500];
	int temp2[cam_height][500];

	int count_1[cam_height];
	int count_2[cam_height];
	for (int i = 0; i < cam_height; i++)
	{
		count_1[i] = 0;
		count_2[i] = 0;
	}
	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 1; j < cam_width-1; j++)
		{
			if ((wrappedPhaseMap.at<float>(i, j - 1) > wrappedPhaseMap.at<float>(i, j)) &&
				(wrappedPhaseMap.at<float>(i, j + 1) > wrappedPhaseMap.at<float>(i, j)))
			{
				temp[i][count_1[i]] = j;//将对应的x值存储
				count_1[i] = count_1[i]+ 1;///行数height,存在多少个条纹阶数
			}
			if ((wrappedPhaseMap.at<float>(i, j - 1) < wrappedPhaseMap.at<float>(i, j)) &&
				(wrappedPhaseMap.at<float>(i, j + 1) < wrappedPhaseMap.at<float>(i, j)))
			{
				temp2[i][count_2[i]] = j;//将对应的x值存储
				count_2[i] = count_2[i]+ 1;///行数height,存在多少个条纹阶数
			}
		}
	}
	int kk;//定义校正后的条纹阶数
	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 1; j < count_1[i] - 1; j++)
		{
			int m1 = temp[i][j-1];
			int m2 = temp[i][j];//第j+1个条纹阶数
			int m3 = temp[i][j+1];
			if ((k[i][m1].x + 1 == k[i][m2].x) && (k[i][m2].x + 1 == k[i][m3].x) )
			{
				int h;
				for (h = 0; h < count_1[i]; h++)
				{
					int y_axis = temp[i][h];//求条纹结束的横坐标，第h+1个
					int y_axis2 = temp[i][h + 1];



					//k[i][y_axis].x = k[i][m3].x - (j - h);
					for (int n = y_axis-1; n < y_axis2-1; n++)
					{
						k[i][n].x = k[i][m2].x - (j - h);
					}
					
				}
				if (h == count_1[i])
				{
					int y_axis3 = temp[i][count_1[i] - 1];//找到最后一个最低点的横坐标
					int y_axis4 = temp2[i][count_2[i] - 1];//找到最后一个最高点的横坐标
					for (int l = y_axis3 - 1; l < y_axis4 - 1; l++)
					{
						k[i][l].x = k[i][m2].x - (j - (count_1[i] - 1));
					}
				}
				break;
			}
		}
	}
//
//输出校正后的条纹阶数
	ofstream fout8("./matlab/xiuzheng_k.txt");
	for (int i = 0; i < cam_width; i++)
	{
		fout8 << k[x][i].x << endl;
	}
	fout8.close();


	/// phase1 
	double phase1_x[1000];
	int count1 = 0;
	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 0; j < cam_width; j++)
		{
			phase1[i][j] = wrappedPhaseMap.at<float>(i, j) + 2 * CV_PI * k[i][j].x;
			
		}
	}
	//for (int i = 3; i < cam_height-3; i++)
	//{
	//	for (int j = 3; j < cam_width-3; j++)
	//	{
	//		for (int x = -3; x <= 3; x++)
	//		{
	//			for (int y = -3; y < 3; y++)
	//			{
	//				phase1_x[count1] = phase1[i + x][j + y];
	//				count1 = count1 + 1;

	//			}
	//		}
	//		sort(phase1_x, phase1_x + count1);
	//		int med = floor(count1 / 2);
	//		if ((phase1[i][j] - phase1_x[med]) > 1)
	//			phase1[i][j] -= 6;
	//		else if ((phase1[i][j] - phase1_x[med]) < -1)
	//			phase1[i][j] += 6;
	//	}
	//}
	//中值滤波
	//cv::Mat img(540, 720, CV_32FC1); 
	cv::Mat img(1024, 1280, CV_32FC1); 
	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 0; j < cam_width; j++)
		{
			img.at<float>(i, j) = phase1[i][j];
		}
	}
	//cv::medianBlur(img, img, 5);

	ofstream fout1("./matlab/gray_line_350.txt");
	ofstream fout2("./matlab/wrapped_line_350.txt");
	ofstream fout3("./matlab/unwrapped_line_350.txt");
	ofstream fout4("./matlab/median.txt");//中值滤波后的解包裹相位图
	ofstream fout5("./matlab/untreated.txt");
	ofstream fout6("./matlab/wrapped_low_350.txt");

	for (int i = 0; i < cam_height; i++)
	{
		for (int j = 0; j < cam_width; j++)
		{
			phase1[i][j] = img.at<float>(i, j);
		}
	}

	for (int i = 0; i < cam_width; i++)
	{
		fout5 << phase1[x][i] << endl;
	}
	
	//cout << "the grey code is used" << endl;
	// get the phase: 解码得到相位值
	
	double win_x[2000];
	const int alpha = 10;
	for (int i = alpha; i < cam_height - alpha; i++)
	{
		for (int j = alpha; j < cam_width - alpha; j++)
		{
			//if (D[i][j] == 1)
			if(shadowmask.at<uchar>(i,j) > 200)
			{
				int count = 0;
				for (int k1 = -alpha; k1 <= alpha; k1++)
				{
					for (int k2 = -alpha; k2 <= alpha; k2++)
					{
						//if (D[i + k1][j + k2] == 1) {
						if (shadowmask.at<uchar>(i+k1, j+k2) > 200) {
							win_x[count] = phase1[i + k1][j + k2];
							count = count + 1;
						}
					}
				}
				sort(win_x, win_x + count);
				int med = floor(count / 2);
				if ((phase1[i][j] - win_x[med]) > PI)
					phase1[i][j] -= 2 * PI;
				else if ((phase1[i][j] - win_x[med]) < -PI)
					phase1[i][j] += 2 * PI;
			}
			Pro_point[i][j] = (phase1[i][j] + dec_config.phase_offset) * dec_config.cycle / (2 * PI) + 1;
		}
	}

	//校正双频解包裹物体边缘的解包裹相位
	for (int i = 0; i < cam_width; i++)
	{
		fout1 << k[x][i].x << endl;
		fout2 << wrappedPhaseMap.at<float>(x, i) << endl;
		fout3 << phase1[x][i] << endl;
		fout4 << img.at<float>(x, i) << endl;
		fout6 << wrappedPhaseMap_low.at<float>(x, i) << endl;
	}
	fout1.close();
	fout2.close();
	fout3.close();
	fout4.close();
	fout6.close();

}
