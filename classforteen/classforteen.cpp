#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void creatKmean(Mat src, Mat &mask)
{
	int width = src.cols;
	int height = src.rows;
	int pix = width * height;
	int cluster = 2;  //分成几类
	Mat labels;   //标记图
	Mat centers;  //初始化簇心

	Mat sumpleData = src.reshape(3, pix);
	Mat km_data;  //聚类数据，浮点型
	sumpleData.convertTo(km_data, CV_32F);
	TermCriteria termcriteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);  //迭代到最大次数为止
	kmeans(km_data, cluster, labels, termcriteria, cluster, KMEANS_PP_CENTERS, centers);  //Kmeans++算法初始化簇心

	for (int i = 0; i < height; i++)  //二值化
	{
		for (int j = 0; j < width; j++)
		{
			if (labels.at<int>(i*width + j) == 0)
				mask.at<uchar>(i, j) = 0;
			else
				mask.at<uchar>(i, j) = 255;
		}
	}
}

int main()
{
	Mat in_src = imread("test.jpg");
	Mat mask;
	mask.create(in_src.size(), CV_8UC1);
	creatKmean(in_src, mask);
	imshow("src", in_src);
	imshow("mask", mask);
	waitKey(0);
}
