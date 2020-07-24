#pragma once
#include <opencv2/opencv.hpp>

std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>>  backprop(cv::Mat x, cv::Mat y, std::vector<cv::Mat> biases, std::vector<cv::Mat> weights);

