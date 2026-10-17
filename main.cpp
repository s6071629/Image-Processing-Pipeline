#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace std;

namespace fs = std::filesystem;

uchar toPixel(float value) {
    int rounded = static_cast<int>(round(value));
    if (rounded < 0) rounded = 0;
    if (rounded > 255) rounded = 255;
    return static_cast<uchar>(rounded);
}

int mirrorIndex(int pos, int size) {
    if (pos < 0) pos = -pos;
    if (pos >= size) pos = 2 * (size - 1) - pos;
    return pos;
}

int clampIndex(int pos, int size) {
    if (pos < 0)
        return 0;
    if (pos >= size)
        return size - 1;
    return pos;
}

cv::Mat myGrayscale(const cv::Mat& colorImage) {
    cv::Mat gray(colorImage.rows,colorImage.cols,CV_8UC1);
    for (int y = 0; y < colorImage.rows; y++) {
        for (int x = 0; x < colorImage.cols; x++) {
            cv::Vec3b pixel = colorImage.at<cv::Vec3b>(y, x);
            float blue = pixel[0];
            float green = pixel[1];
            float red = pixel[2];
            float value = 0.299f * red + 0.587f * green + 0.114f * blue;
            gray.at<uchar>(y, x) = toPixel(value);
        }
    }
    return gray;
}

vector<float> makeGaussianWeights(int kernelSize, double sigma) {
    int radius = kernelSize / 2;
    vector<float> weights(kernelSize);
    float total = 0.0f;
    for (int offset = -radius; offset <= radius; offset++) {
        float w = exp(-(offset * offset) / (2.0f * sigma * sigma));
        weights[offset + radius] = w;
        total += w;
    }
    for (float& w : weights) {
        w /= total;
    }
    return weights;
}

cv::Mat myGaussianBlur(const cv::Mat& gray,int kernelSize,double sigma) {
    int radius = kernelSize / 2;
    vector<float> weights = makeGaussianWeights(kernelSize, sigma);
    cv::Mat horizontal(gray.rows,gray.cols,CV_32FC1);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            float sum = 0.0f;
            for (int offset = -radius; offset <= radius; offset++) {
                int neighbourX = mirrorIndex(x + offset,gray.cols);
                sum += weights[offset + radius] * gray.at<uchar>(y,neighbourX);
            }
            horizontal.at<float>(y, x) = sum;
        }
    }
    cv::Mat result(gray.rows,gray.cols,CV_8UC1);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            float sum = 0.0f;
            for (int offset = -radius; offset <= radius; offset++) {
                int neighbourY = mirrorIndex( y + offset, gray.rows);
                sum += weights[offset + radius] * horizontal.at<float>(neighbourY,x);
            }
            result.at<uchar>(y, x) = toPixel(sum);
        }
    }
    return result;
}

cv::Mat myMedianFilter(const cv::Mat& gray) {
    cv::Mat result(gray.rows,gray.cols,CV_8UC1);
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            vector<uchar> neighbours;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = clampIndex( y + dy,gray.rows);
                    int nx = clampIndex(x + dx, gray.cols);
                    neighbours.push_back(gray.at<uchar>(ny, nx));
                }
            }
            sort(neighbours.begin(),neighbours.end());
            result.at<uchar>(y, x) =  neighbours[4];
        }
    }
    return result;
}

cv::Mat mySobel(const cv::Mat& gray) {
    const int sobelX[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
    const int sobelY[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
    cv::Mat result(gray.rows,gray.cols,CV_8UC1);

    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            int gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = mirrorIndex(y + dy,gray.rows);
                    int nx = mirrorIndex(x + dx,gray.cols);
                    int brightness = gray.at<uchar>(ny,nx);
                    gx += sobelX[dy + 1][dx + 1] * brightness;
                    gy += sobelY[dy + 1][dx + 1] *brightness;
                }
            }
            float strength = sqrt(static_cast<float>(gx * gx + gy * gy));
            result.at<uchar>(y, x) = toPixel(strength);
        }
    }
    return result;
}

cv::Mat opencvGrayscale(const cv::Mat& colorImage) {
    cv::Mat gray;
    cv::cvtColor(colorImage,gray,cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat opencvGaussianBlur(const cv::Mat& gray,int kernelSize,double sigma) {
    cv::Mat blurred;
    cv::GaussianBlur(gray,blurred,cv::Size(kernelSize, kernelSize),sigma,sigma);
    return blurred;
}

cv::Mat opencvMedianFilter(const cv::Mat& gray) {
    cv::Mat filtered;
    cv::medianBlur(gray,filtered,3);
    return filtered;
}

cv::Mat opencvSobel(const cv::Mat& gray) {
    cv::Mat gradX, gradY, strength, result;
    cv::Sobel(gray,gradX,CV_32F,1,0,3);
    cv::Sobel(gray,gradY,CV_32F,0,1,3);
    cv::magnitude(gradX,gradY,strength);
    strength.convertTo(result,CV_8U);
    return result;
}

double calculateMSE(const cv::Mat& imageA,const cv::Mat& imageB) {
    double sumOfSquares = 0.0;
    for (int y = 0; y < imageA.rows; y++) {
        for (int x = 0; x < imageA.cols; x++) {
            double difference = static_cast<double>(imageA.at<uchar>(y, x)) - 
                static_cast<double>(imageB.at<uchar>(y, x));
            sumOfSquares += difference * difference;
        }
    }
    return sumOfSquares /(imageA.rows * imageA.cols);
}

double calculatePSNR(double mse) {
    if (mse == 0.0) {
        return numeric_limits<double>::infinity();
    }
    return 10.0 * log10((255.0 * 255.0) / mse);
}

int largestDifference(const cv::Mat& imageA, const cv::Mat& imageB) {
    cv::Mat difference;
    cv::absdiff(imageA,imageB,difference);
    double maxValue = 0;
    cv::minMaxLoc(difference,nullptr,&maxValue);
    return static_cast<int>(maxValue);
}

double measureTimeMs( const function<void()>& task, int repeats = 10) {
    task();
    auto start = chrono::steady_clock::now();

    for (int i = 0; i < repeats; i++) {
        task();
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double, milli> total = end - start;
    return total.count() / repeats;
}

struct ComparisonResult {
    string name;
    double mse, psnr, myTimeMs, opencvTimeMs;
    int maxDiff;
};

ComparisonResult compare(const string& name, const cv::Mat& mine,
    const cv::Mat& reference, double myTimeMs, double opencvTimeMs) {
    ComparisonResult r;
    r.name = name;
    r.mse = calculateMSE( mine, reference);
    r.psnr = calculatePSNR(r.mse);
    r.maxDiff = largestDifference( mine, reference);
    r.myTimeMs = myTimeMs;
    r.opencvTimeMs = opencvTimeMs;
    return r;
}

void printResult(const ComparisonResult& r) {
    cout << left << r.name << right << fixed;
    cout << " MSE = " << setprecision(4) << r.mse;
    if (isinf(r.psnr)) {
        cout << "PSNR = " << "inf";
    } else {
        cout << "PSNR = " << setprecision(2)<< r.psnr << " dB";
    }
    cout << " maxDiff = " << r.maxDiff << "  mine = "
         << setprecision(3) << r.myTimeMs << " ms "
         << "opencv = " << r.opencvTimeMs << " ms " << "  slowdown = x"
         << setprecision(1) << (r.myTimeMs / r.opencvTimeMs) << "\n";
}

void saveResultsCsv(const string& path, const vector<ComparisonResult>& results) {
    ofstream file(path);
    if (!file.is_open()) {
        cerr << "Warning: could not open " << path << " for writing.\n";
        return;
    }

    file << "operation," << "MSE," << "PSNR(dB)," << "max_pixel_diff," << "my_time_ms,"
         << "opencv_time_ms," << "slowdown\n";
    for (const ComparisonResult& r : results) {
        file << r.name << "," << r.mse << ","
             << (isinf(r.psnr) ? "inf" : to_string(r.psnr))
             << "," << r.maxDiff << "," << r.myTimeMs << ","
             << r.opencvTimeMs << ","
             << (r.myTimeMs / r.opencvTimeMs) << "\n";
    }
    file.close();
}

void saveImage(const fs::path& folder, const string& fileName, const cv::Mat& image) {
    fs::path fullPath = folder / fileName;
    if (!cv::imwrite(fullPath.string(),image)) {
        cerr << "Warning: could not save "<< fullPath << "\n";
    }
}

int main (int argc,char** argv) {
    if (argc < 1) {
        cerr << "Usage: " << argv[0] << " <image_path>";
        return 1;
    }

    string imagePath =argv[1];

    fs::path outputFolder = "output";

    fs::create_directories(outputFolder);

    cv::Mat colorImage =cv::imread(imagePath,cv::IMREAD_COLOR);

    if (colorImage.empty()) {
       cerr << "Error: could not read image '" << imagePath;
        return 1;
    }

    const int kernelSize = 5;
    const double sigma = 1.0;

    cv::Mat grayMine =myGrayscale(colorImage);
    cv::Mat gaussMine =myGaussianBlur(grayMine,kernelSize,sigma);
    cv::Mat medianMine = myMedianFilter(grayMine);
    cv::Mat sobelMine = mySobel(grayMine);

    cv::Mat grayCv = opencvGrayscale(colorImage);
    cv::Mat gaussCv = opencvGaussianBlur(grayCv,kernelSize,sigma);
    cv::Mat medianCv = opencvMedianFilter(grayCv);
    cv::Mat sobelCv = opencvSobel(grayCv);

    vector<ComparisonResult> results;
    results.push_back(
        compare(
            "Grayscale",grayMine,grayCv,
            measureTimeMs([&] {myGrayscale(colorImage);}),
            measureTimeMs([&] {opencvGrayscale(colorImage);})
        )
    );
    results.push_back(
        compare(
            "Gaussian 5x5",gaussMine,gaussCv,
            measureTimeMs([&] {myGaussianBlur(grayMine,kernelSize,sigma);}),
            measureTimeMs([&] {opencvGaussianBlur(grayCv,kernelSize,sigma);})
        )
    );
    results.push_back(
        compare(
            "Median 3x3",medianMine,medianCv,
            measureTimeMs([&] {myMedianFilter(grayMine);}),
            measureTimeMs([&] {opencvMedianFilter(grayCv);})
        )
    );
    results.push_back(
        compare(
            "Sobel",sobelMine,sobelCv,
            measureTimeMs([&] {mySobel(grayMine);}),
            measureTimeMs([&] {opencvSobel(grayCv);})
        )
    );
    cout << "Comparison Results:\n";
    for (const ComparisonResult& r : results) {
        printResult(r);
    }
    saveImage(outputFolder,"0_original.png",colorImage);

    saveImage(outputFolder,"gray_mine.png",grayMine);

    saveImage(outputFolder,"gray_opencv.png",grayCv);

    saveImage(outputFolder,"gaussian_mine.png",gaussMine);

    saveImage(outputFolder,"gaussian_opencv.png",gaussCv);

    saveImage(outputFolder,"median_mine.png",medianMine);

    saveImage(outputFolder,"median_opencv.png",medianCv);

    saveImage(outputFolder,"sobel_mine.png",sobelMine);

    saveImage(outputFolder,"sobel_opencv.png",sobelCv);
    cv::Mat sobelDifference;
    cv::absdiff(sobelMine,sobelCv,sobelDifference);
    saveImage(outputFolder,"5_sobel_difference.png",sobelDifference * 50);
    saveResultsCsv((outputFolder / "results.csv").string(),results);
    return 0;
}