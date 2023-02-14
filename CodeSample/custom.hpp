#ifndef __CUSTOM_H__
#define __CUSTOM_H__
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
using namespace cv;

bool findChessboardCornersCustom(InputArray image_, Size pattern_size, OutputArray corners_, int flags);

#endif //__CUSTOM_H__s