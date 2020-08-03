#include "backprop.h"
#include <opencv2/opencv.hpp>
#include <boost/iterator/zip_iterator.hpp>

//return 1.0 / (1.0 + np.exp(-z))
auto sigmoid(cv::Mat in) {
    //auto z = input.clone();
    //for (int i = 0; i < z.rows; i++)//through each row in Mat M
    //    for (int j = 0; j < z.cols; j++)//through each column
    //        z.at<double>(i, j) = 1.0/(1.0 + std::exp(-z.at<double>(i, j)));
    //return z;
    cv::exp(-in, in);
    return 1 / (1 + in);
}

auto sigmoid_prime(cv::Mat z) {
    cv::Mat ret = sigmoid(z.clone());
    ret = ret.mul(cv::Mat(z.size(),CV_64FC1,cv::Scalar(1.0)) - sigmoid(z.clone()));
    return ret;
}

auto cost_derivative(cv::Mat activation, cv::Mat yy) {
    cv::Mat y;
    yy.convertTo(y, CV_64FC1);
    auto a = activation.clone();
    a = a - y;
    return a;
}

std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>>  backprop(cv::Mat x, cv::Mat y, std::vector<cv::Mat> biases, std::vector<cv::Mat> weights) {

    // Y DON'T use
    std::vector<cv::Mat> nabla_b;
    std::vector<cv::Mat> nabla_w;

    nabla_b.resize(biases.size());
    nabla_w.resize(weights.size());

    //for (auto x : biases) {
    //    nabla_b.push_back(cv::Mat(x.size(), CV_64F, double(0)));
    //}
    //for (auto x : weights) {
    //    nabla_w.push_back(cv::Mat(x.size(), CV_64F, double(0)));
    //}
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
        zs.push_back(z.clone());
        activation = sigmoid(z);

        activations.push_back(activation.clone());
    }

    auto m1 = cost_derivative(activations[activations.size() - 1], y);
    auto m2 = sigmoid_prime(zs[zs.size() - 1]);
    cv::Mat delta = m1.mul(m2);

    nabla_b[nabla_b.size() - 1] = delta;
    cv::Mat t;
    cv::transpose(activations[activations.size() - 2], t);
    cv::Mat temp = delta* t;
    nabla_w[nabla_w.size() - 1] = temp;

    for (int l = 2; l < 3; l++) {
        auto z = zs[zs.size() - l];
        auto sp = sigmoid_prime(z);
        //delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
        cv::Mat t;
        cv::transpose(weights[weights.size()-l + 1], t);
        delta = (t * delta).mul(sp);
        nabla_b[nabla_b.size() - l] = delta.clone();
        
        cv::transpose(activations[activations.size() -l - 1], t);
        cv::Mat tt = delta * t;
        nabla_w[nabla_w.size() -l] = tt.clone();



    }
    return {nabla_b, nabla_w};
}

