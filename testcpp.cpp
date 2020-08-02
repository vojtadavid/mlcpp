#include <iostream>
#include <opencv2/opencv.hpp>



int main() {
	cv::Mat A = cv::Mat::eye(3, 5, CV_32FC1);
	A.at<float>(1, 1) = 123.456789096765;
	std::cout << A;
}