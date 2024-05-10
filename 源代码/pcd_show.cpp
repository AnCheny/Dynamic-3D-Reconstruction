#include "iostream"
#include <pcl/io/pcd_io.h>     //pcd��д�ļ�
#include <pcl/point_types.h>   //�������ļ�
#include <pcl/point_cloud.h>   
#include <pcl/visualization/cloud_viewer.h>
using namespace std;

extern "C" int pcd_show()
{
	//����һ������ָ��
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//���ص��Ʋ��ж��Ƿ���سɹ�
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("E:\\vs_engineering\\structured_light\\structured_light\\test_pcd_2.pcd", *cloud) == -1)
	{
		PCL_ERROR("could not read file test.pcd\n");
		return(-1);
	}
	cout << cloud->points.size() << endl;
	//----------------------------------------------------------------------------
		//����һ�����ƿ��ӻ�����
	pcl::visualization::CloudViewer view("cloud_viewer");
	//���ӻ�����
	view.showCloud(cloud);
	while (!view.wasStopped())   //�ȴ�
	{

	}
	system("pause");
	return 0;
}