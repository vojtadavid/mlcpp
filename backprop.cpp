#include "backprop.h"
#include <opencv2/opencv.hpp>
#include <boost/iterator/zip_iterator.hpp>

std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>>  backprop(cv::Mat x, cv::Mat y, std::vector<cv::Mat> biases, std::vector<cv::Mat> weights) {

    // Y DON'T use
    std::vector<cv::Mat> nabla_b;
    std::vector<cv::Mat> nabla_w;

    for (auto x : biases) {
        nabla_b.push_back(cv::Mat(x.size(), CV_64F, double(0)));
    }
    for (auto x : weights) {
        nabla_w.push_back(cv::Mat(x.size(), CV_64F, double(0)));
    }
    auto activation = cv::Mat(1, 784, CV_64F);
    activation = x.reshape(1, 784);
    activation.convertTo(activation, CV_64F);
    activation = activation / 255.0;
    std::vector<cv::Mat> activations;
    activations.push_back(activation.clone());
    std::vector<cv::Mat> zs;

    for (int i = 0; i < biases.size(); i++) {
        auto b = biases[i];
        auto w = weights[i];
        cv::Mat z = w * activation + b;

        std::cout << z << '\n';
    }

    //std::for_each(
    //    boost::make_zip_iterator(
    //        boost::make_tuple(std::begin(biases), std::begin(weights))
    //    ),
    //    boost::make_zip_iterator(
    //        boost::make_tuple(std::end(biases), std::end(weights))
    //    ),
    //    [&activation](auto tuple) {
    //        auto b = tuple.get<0>();
    //        auto w = tuple.get<1>();


    //        cv::Mat z = w * activation + b;

    //        std::cout << z << '\n';

    //        z;
    //    }
    //);




    return std::make_tuple(nabla_b, nabla_w);
}

