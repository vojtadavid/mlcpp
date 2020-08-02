// mlcpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/algorithm/string.hpp>

#include "backprop.h"

#include <filesystem>
namespace fs = std::filesystem;


template<typename T>
struct Network 
{
    int num_layers;
    std::vector<int> sizes;
    std::vector<cv::Mat> biases;
    std::vector<cv::Mat> weights;
    std::vector<std::tuple<cv::Mat, int>> training_data;
    std::vector<std::tuple<cv::Mat, int>> test_data;




    Network(std::vector<int> sizes) {
        this->sizes = sizes;
        auto it2 = sizes.begin();

        std::random_device rd{};
        std::mt19937 generator{ rd() };
        std::normal_distribution<> dis{ 0.0,1.0 };

        for (auto it = ++sizes.begin(); it != sizes.end(); it++) {
            //std::vector<T> bias(*it);
            cv::Mat bias = cv::Mat(*it,1, CV_64FC1);

            auto gen = [&dis,&generator]() {return dis(generator); };
            std::generate(bias.begin<double>(), bias.end<double>(), gen);
            biases.push_back(bias);

            cv::Mat w(*it, *it2, CV_64FC1);
            std::generate(w.begin<double>(), w.end<double>(), gen);
            weights.push_back(w);

            it2++;

        }


        auto images = loadTrainImages("./../train-images-idx3-ubyte/train-images.idx3-ubyte");
        auto labels = loadTrainLabels("./../train-labels-idx1-ubyte/train-labels.idx1-ubyte");

        auto test_images = loadTrainImages("./../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
        auto train_labels= loadTrainLabels("./../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

        

        auto l = labels.begin();
        for (auto& i : images) {
            training_data.push_back(std::make_tuple(i, *l));
            l++;
        }

        l = train_labels.begin();
        for (auto& i : test_images) {
            test_data.push_back(std::make_tuple(i, *l));
            l++;
        }
    }

    void saveMat(std::string filename, cv::Mat M) {
        std::ofstream outfile;
        outfile.open(filename,std::ios::out);
        outfile << M.rows << " " << M.cols << '\n';
        for (int i = 0; i < M.rows; i++)
        {
            for (int j = 0; j < M.cols; j++) {
                outfile << M.at<T>(cv::Point(j,i)) << ' ';
            }
            outfile << "\n";
        }
    }

    cv::Mat readMat(std::string filename) {
        std::ifstream infile;
        infile.open(filename, std::ios::in);
        std::vector < std::vector<T>> input;
        std::string str;

        int rows, cols;
        infile >> rows >> cols;
        std::getline(infile, str);
        auto A = cv::Mat(rows, cols, CV_64FC1);
        int r = 0;
        while (std::getline(infile, str)) {
            std::vector<std::string> results;
            boost::split(results, str, [](char c) {return c == ' '; });
            for (auto c = 0; c < cols; c++) {
                A.at<double>(cv::Point(c, r)) = std::stod(results[c]);
            }
            r++;
        }
        return A;
    }

    void savedata() {
        fs::path p1 = "./data/";
        if (!fs::exists(p1)) {
            fs::create_directory("./data/");
        }
        int b = 0;
        for (auto x : biases) {
            saveMat("./data/b"+std::to_string(b), x);
            b++;
        }
        int w = 0;
        for (auto x : weights) {
            saveMat("./data/w"+std::to_string(w), x);
            w++;
        }
    }

    void loaddata() {
        fs::path p1 = "./data/";
        biases.clear();
        weights.clear();
        for (auto& p : fs::directory_iterator("data")) {
            std::cout << p.path() << " " << p.path().filename() << '\n';
            auto name = p.path().filename().generic_string();
            if (name[0]=='w') {
                auto A = readMat(p.path().generic_string());
                weights.push_back(A);

            }
            if (name[0] == 'b') {
                auto A = readMat(p.path().generic_string());
                biases.push_back(A);

            }

        }

    }

    auto loadTrainImages(std::string path) {
        std::ifstream istrm(path, std::ios::binary);
        int magic, no_images, rows, colums;
        istrm.read((char*)&magic, 4);
        istrm.read((char*)&no_images, 4);
        istrm.read((char*)&rows, 4);
        istrm.read((char*)&colums, 4);

        magic = reverseInt(magic);
        no_images = reverseInt(no_images);
        rows = reverseInt(rows);
        colums = reverseInt(colums);

        //printf("magic %d\nnumber of images %d\nrows %d\ncolums %d\n", magic, no_images, rows, colums);

        std::vector<cv::Mat> images;
        images.reserve(no_images);
        for (int n = 0; n < no_images; n++) {
            cv::Mat A(rows, colums, CV_8UC1);
            istrm.read((char*)A.data, rows * colums);
            images.push_back(A);
            //cv::imshow("A", A);
            //cv::waitKey();
        }
        return images;
    }

    auto loadTrainLabels(std::string path) {
        std::ifstream istrm(path, std::ios::binary);
        if (!istrm.is_open()) {
            throw std::runtime_error("can't open file " + path);
        }

        int no_labels, magic;
        istrm.read((char*)&magic, 4);
        istrm.read((char*)&no_labels, 4);

        no_labels = reverseInt(no_labels);

        std::vector<uint8_t> labels(no_labels);
        istrm.read((char*)labels.data(), no_labels);
        return labels;
    }

    cv::Mat sigmoid(cv::Mat& in) {
        //cv::Mat result = in.clone();
        cv::exp(-in, in);
        return 1 / (1 + in);


    }

    void update_mini_batch(int start, int end, double eta) {
        std::vector<cv::Mat> nabla_b;
        std::vector<cv::Mat> nabla_w;
        
        for (auto x : biases) {
            nabla_b.push_back(cv::Mat(x.size(),CV_64F, double(0)));
        }
        for (auto x : weights) {
            nabla_w.push_back(cv::Mat(x.size(), CV_64F, double(0)));
        }

        for (int i = start; i < end; ++i) {
            auto label = cv::Mat(10, 1, CV_32SC1);
            label = 0;
            int l = std::get<1>(training_data[i]);
            label.at<int>(l,0) = 1;
            auto [delta_nabla_b, delta_nabla_w] = backprop(std::get<0>(training_data[i]), label,biases,weights);

            for (int j = 0; j < delta_nabla_b.size(); j++) {
                nabla_b[j] = nabla_b[j] + delta_nabla_b[j];
            }

            for (int j = 0; j < delta_nabla_w.size(); j++) {
                nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
            }
        }


        for (int i = 0; i < this->weights.size(); i++) {
            this->weights[i] = this->weights[i] - (eta/(end-start))*nabla_w[i];
        }

        for (int i = 0; i < this->biases.size(); i++) {
            this->biases[i] = this->biases[i] - (eta / (end - start)) * nabla_b[i];
        }

        cv::FileStorage fs;
        fs.open("w.yml", cv::FileStorage::WRITE);
        fs.write("ww", weights[0]);
        cv::rou
        fs.release();



    }

    void SGD(int epochs = 30, int mini_batch_size = 10, double eta = 3.0) {
        std::random_device rd;
        std::mt19937 g(rd());
        for (int j = 0; j < epochs; j++) {
            //std::shuffle(training_data.begin(), training_data.end(), g);
            
            std::vector<std::tuple<cv::Mat, int>> mini_batches;
            for (int b = 0; b < training_data.size(); b += mini_batch_size) {
                update_mini_batch(b, b + mini_batch_size, eta);
            }

            if (!test_data.empty()) {
                evaluate();
            }

        }
    }

    void evaluate() {
        for (auto& td: this->test_data) {
            auto label = cv::Mat(10, 1, CV_32SC1);
            label = 0;
            int l = std::get<1>(td);
            label.at<int>(l, 0) = 1;

            feedforward(std::get<0>(td));
        }
    }

    auto feedforward(cv::Mat a) {
        auto A = cv::Mat(a.reshape(1, 784));
        A.convertTo(A, CV_64FC1);
        A = A / 255.0;
        for (int i = 0; i < this->weights.size(); i++) {
            cv::Mat t = weights[i] * A + biases[i];
            A = sigmoid(t);
        }
        return A;
    }

};

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



int main()
{
  
    Network<double> net({ 784,30,10 });
    //net.savedata();
    net.loaddata();
    net.SGD();




}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
