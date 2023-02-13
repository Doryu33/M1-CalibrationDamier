#ifndef __CUSTOM_H__
#define __CUSTOM_H__
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
using namespace cv;

template<typename ArrayContainer>
static void icvGetIntensityHistogram256(const Mat& img, ArrayContainer& piHist);

template<int iWidth_, typename ArrayContainer>
static void icvSmoothHistogram256(const ArrayContainer& piHist, ArrayContainer& piHistSmooth, int iWidth = 0);

static void icvBinarizationHistogramBased(Mat & img);

template<typename ArrayContainer>
static void icvGradientOfHistogram256(const ArrayContainer& piHist, ArrayContainer& piHistGrad);

bool findChessboardCornersCustom(InputArray image_, Size pattern_size, OutputArray corners_, int flags);

#endif //__CUSTOM_H__s