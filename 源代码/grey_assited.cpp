

#include "..\\include\\opencv2\\para_config.h"
#include "..\\include\\opencv2\\grey_assited.h"

void decode_gp_single(bool** D, cv::Mat& wrappedPhaseMap, double** Pro_point, int image_count, char* path/*, double** phase1*/)//
{


	int grey_col = ceil(log2(double(pro_width) / double(dec_config.cycle)));
	int w_i = ceil(double(pro_width) / double(dec_config.cycle));
	int N_w = ceil(log2(w_i));

	double offset_w = floor((pow(2, N_w) - w_i) / 2);

	//Mat Min_pixel = 255 * Mat::ones(cam_height, cam_width, CV_64FC1);
	//Mat Max_pixel = Mat::zeros(cam_height, cam_width, CV_64FC1);

	char path1[100];
	char path2[100];
	char path3[100];
	char path4[100];
	char sub_name1[20];
	char sub_name2[20];
	char sub_name3[20];
	char sub_name4[20];
	vector<cv::Mat> image_grey(6);
	vector<cv::Mat> image_phase(3);
	//Mat Min_pixel = 255 * Mat::ones(cam_height, cam_width, CV_64FC1);
	//Mat Max_pixel = Mat::zeros(cam_height, cam_width, CV_64FC1);
	cv::Mat Max_pixel, Min_pixel;

	//* 读取拍摄的图像，存入到image_phase中，用于解码
	for (int i = 0; i < 11 /*grey_col + dec_config.phase_num + 2*/; i++)	//	gray pattern and white,black 6 +3+2
	{
		int i_x = 2 * i + 1;//1,3,5,7,9,11,13,15,17,19,21,23
		int i_y = 2 * (i + 1);//2,4,6,8,10,12,14,16,18,20,22
		int temp_phase = i % 3; // 0,1,2
		int temp_grey = i % 6;//0,1,2,3,4,5
		// grey code 
		if (i < 6)
		{
			sprintf(sub_name1, "000%d.bmp", /*image_count, */(i));
			strcpy(path1, path);// path = ./decode_image/
			strcat(path1, sub_name1);
			cv::Mat temp1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);//读入灰度图
			temp1.convertTo(temp1, CV_64FC1);

			image_grey.insert(image_grey.begin() + i, temp1);
			//image_grey[temp_grey]=temp1.clone();
		}
		
		else if(7<i && i<11)
		{ 
			//phase img
			sprintf(sub_name2, "000%d.bmp", /*image_count*/ (i));
			strcpy(path2, path);// path = ./decode_image/
			strcat(path2, sub_name2);
			cv::Mat temp2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);//读入灰度图
			temp2.convertTo(temp2, CV_64FC1);

			image_phase.insert(image_phase.begin() + (i-8), temp2);
		}
	}

		// Min_pixel
		//sprintf(sub_name3, "%d_5.bmp", image_count);
		sprintf(sub_name3, "0006.bmp");
		strcpy(path3, path);// path = ./decode_image/
		strcat(path3, sub_name3);
		cv::Mat temp3 = cv::imread(path3, cv::IMREAD_GRAYSCALE);//读入灰度图
		temp3.convertTo(temp3, CV_64FC1);
		Min_pixel = temp3;

		// Max_pixel
		//sprintf(sub_name4, "%d_3.bmp", image_count);
		sprintf(sub_name4, "0007.bmp");
		strcpy(path4, path);// path = ./decode_image/
		strcat(path4, sub_name4);
		cv::Mat temp4 = cv::imread(path4, cv::IMREAD_GRAYSCALE);//读入灰度图
		temp4.convertTo(temp4, CV_64FC1);
		Max_pixel = temp4;//灰度值的范围是0-255

		//for (int i = 0; i < cam_height; i++)
		//{
		//	for (int j = 0; j < cam_width; j++)
		//	{
		//		for (int k = 0; k < 2 * dec_config.grey_num + 2; k++)
		//		{
		//			if (Max_pixel.at<float>(i, j) < image_grey[k].at<float>(i, j))
		//				Max_pixel.at<float>(i, j) = image_grey[k].at<float>(i, j);
		//			if (Min_pixel.at<float>(i, j) > image_grey[k].at<float>(i, j))
		//				Min_pixel.at<float>(i, j) = image_grey[k].at<float>(i, j);
		//		}
		//	}
		//}
		double time0 = static_cast<double>(cv::getTickCount());

		bool*** dec_img = new bool** [cam_height];
		cv::Point2d** A = new cv::Point2d * [cam_height];
		for (int i = 0; i < cam_height; i++)
		{
			dec_img[i] = new bool* [cam_width];
			A[i] = new cv::Point2d[cam_width];
			for (int j = 0; j < cam_width; j++)
			{
				dec_img[i][j] = new bool[grey_col];
			}
		}
		cv::Mat D_temp = Max_pixel - Min_pixel;

		cv::Mat D_temp8;
		D_temp.convertTo(D_temp8, CV_8U, 255, 128);
		cv::imwrite("E:\\vs_engineering\\structured_light\\strctured_light_revised\\structured_light\\out\\OutputWrappedPhase\\D_temp.png", D_temp8);

		for (int i = 0; i < cam_height; i++)
		{
			for (int j = 0; j < cam_width; j++)
			{
				if ((D_temp.at<double>(i, j)) > 30)
					D[i][j] = 1;
				else
					D[i][j] = 0;
			}
		}
		// get grey code：解码获得相应的格雷码
		for (int k = 0; k < grey_col; k++)
		{
			for (int i = 0; i < cam_height; i++)
			{
				for (int j = 0; j < cam_width; j++)
				{
					/*Mat temp;
					temp.push_back(image_grey.begin() + k);*/
					double th = (D_temp.at<double>(i, j) == 0 ? 0 : ((image_grey[k].at<double>(i, j) - Min_pixel.at<double>(i, j)) / D_temp.at<double>(i, j)));
					if (th < dec_config.threshold)
					{
						dec_img[i][j][k] = 0;
					}
					else
					{
						dec_img[i][j][k] = 1;
					}
				}
			}
		}
		for (int k = 0; k < grey_col; k++)
		{
			for (int i = 0; i < cam_height; i++)
			{
				for (int j = 0; j < cam_width; j++)
				{
					if (k == 0) {
						A[i][j].x = float(dec_img[i][j][k]) * pow(2, grey_col - 1) - offset_w;
					}
					else
					{
						dec_img[i][j][k] = dec_img[i][j][k - 1] ^ dec_img[i][j][k];
						A[i][j].x += float(dec_img[i][j][k]) * pow(2, grey_col - k - 1);
					}
				}
			}
		}
		
		ofstream fout1("./matlab/gray_line_450.txt");
		ofstream fout2("./matlab/wrapped_line_450.txt");

		for (int i = 0; i < cam_width; i++)
		{
			fout1 << A[360][i].x << endl;
			fout2 << wrappedPhaseMap.at<float>(380, i) << endl;
		}
		fout1.close();
		fout2.close();
		//cout << "the grey code is used" << endl;
		// get the phase: 解码得到相位值
		cv::Point2d** p_c = new cv::Point2d * [cam_height];
		double** phase1 = new double* [cam_height];
		for (int i = 0; i < cam_height; i++)
		{
			phase1[i] = new double[cam_width];
			p_c[i] = new cv::Point2d[cam_width];
			memset(p_c[i], 0, sizeof(cv::Point2d) * cam_width);
		}
		for (int i = 0; i < cam_height; i++)
		{
			for (int j = 0; j < cam_width; j++)
			{
				phase1[i][j] = -wrappedPhaseMap.at<float>(i,j)+A[i][j].x * 2 * PI;
			}
		}
		double win_x[441];
		for (int i = 10; i < cam_height - 10; i++)
		{
			for (int j = 10; j < cam_width - 10; j++)
			{
				if (D[i][j] == 1)
				{
					int count = 0;
					for (int k1 = -10; k1 <= 10; k1++)
					{
						for (int k2 = -10; k2 <= 10; k2++)
						{
							if (D[i + k1][j + k2] == 1) {
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
	
}
