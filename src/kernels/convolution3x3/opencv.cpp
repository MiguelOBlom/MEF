#include "util.h"

#include <opencv2/opencv.hpp>

long unsigned int N = 2 * trueX * trueX + 16;

void experiment (float * Data) {
    cv::setNumThreads(1);
    
    float *C = &Data[0];
    float *A = &C[16];
    float *B = &A[trueX * trueX];

    cv::Mat Amat(trueX, trueX, CV_32F, A);
    cv::Mat Bmat(trueX, trueX, CV_32F, B);
    cv::Mat Cmat(3, 3, CV_32F, C);

    cv::Mat AmatROI = Amat(cv::Range(1, trueX - 1), cv::Range(1, trueX - 1));
    cv::Mat BmatROI = Bmat(cv::Range(1, trueX - 1), cv::Range(1, trueX - 1));

    cv::filter2D(AmatROI, BmatROI, CV_32F, Cmat);    
}