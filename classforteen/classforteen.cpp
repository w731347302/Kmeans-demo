#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void creatKmean(Mat src, Mat &mask)
{
	int width = src.cols;
	int height = src.rows;
	int pix = width * height;
	int cluster = 2;
	Mat labels;
	Mat centers;

	Mat sumpleData = src.reshape(3, pix);
	Mat km_data;
	sumpleData.convertTo(km_data, CV_32F);
	TermCriteria termcriteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(km_data, cluster, labels, termcriteria, cluster, KMEANS_PP_CENTERS, centers);

	vector<uchar> fg = { 0,255 };
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
			mask.at<uchar>(i, j) = fg[labels.at<int>(i*width + j)];
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