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
        

        auto l = labels.begin();
        for (auto& i : images) {
            training_data.push_back(std::make_tuple(i, *l));
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

    auto feedforward(cv::Mat input) {
        auto in = input.clone();
        auto w = weights.begin();
        auto b = biases.begin();

        for (auto it : weights) {
            in = ( (*w) * in) + *b;
            in = sigmoid(in);
            w++;
            b++;
        }
        std::cout << in << "\n";
        return in;
    }
    void backprop(cv::Mat x,cv::Mat y) {
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

        std::for_each(
            boost::make_zip_iterator(
                boost::make_tuple(std::begin(biases), std::begin(weights))
            ),
            boost::make_zip_iterator(
                boost::make_tuple(std::end(biases), std::end(weights))
            ),
            [&activation](auto tuple) {
                auto b = tuple.get<0>();
                auto w = tuple.get<1>();

                std::cout << b << '\n';
                std::cout << w << '\n';
                std::cout << activation << '\n';


                cv::Mat z = w*activation+b;
                
                std::cout << z << '\n';
            }
        );


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
            backprop(std::get<0>(training_data[i]), std::get<0>(training_data[i]));
        }



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
        }
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
