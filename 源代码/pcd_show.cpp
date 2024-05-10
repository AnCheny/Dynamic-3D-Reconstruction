#include "iostream"
#include <pcl/io/pcd_io.h>     //pcd读写文件
#include <pcl/point_types.h>   //点类型文件
#include <pcl/point_cloud.h>   
#include <pcl/visualization/cloud_viewer.h>
using namespace std;

extern "C" int pcd_show()
{
	//定义一个点云指针
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//加载点云并判定是否加载成功
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("E:\\vs_engineering\\structured_light\\structured_light\\test_pcd_2.pcd", *cloud) == -1)
	{
		PCL_ERROR("could not read file test.pcd\n");
		return(-1);
	}
	cout << cloud->points.size() << endl;
	//----------------------------------------------------------------------------
		//定义一个点云可视化对象
	pcl::visualization::CloudViewer view("cloud_viewer");
	//可视化点云
	view.showCloud(cloud);
	while (!view.wasStopped())   //等待
	{

	}
	system("pause");
	return 0;
}