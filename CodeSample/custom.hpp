#ifndef __CUSTOM_H__
#define __CUSTOM_H__
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stack>

using namespace cv;

struct ImageData
{
  std::string FileName;
  bool mireTrouvee;
  int nbCarresMire;
  int nbCarresDetectes;
  double moyenneLongueurCote_Pixels;
  double medianeLongueurCote_Pixels;
  double minLongueurCote_Pixels;
  double maxLongueurCote_Pixels;
  double ecartTypeLongueurCote_Pixels;
  int nombreCarresDetectesSansExtremes;
  double moyenneLongueurCoteSansExtremes_Pixels;
  double medianeLongueurCoteSansExtremes_Pixels;
  double minLongueurCoteSansExtremes_Pixels;
  double maxLongueurCoteSansExtremes_Pixels;
  double ecartTypeLongueurCoteSansExtremes_Pixels;
};

bool compareByX(const Point& p1, const Point& p2);

bool compareByX2(const Point& p1, const Point& p2);

std::pair<Point, Point> getSmallestXPoints(const std::vector<Point>& points);

std::pair<Point, Point> getBiggestXPoints(const std::vector<Point>& points);

static void icvBinarizationHistogramBased(Mat & img);

static void icvGetQuadrangleHypothesesCustom(const std::vector<std::vector< cv::Point > > & contours, const std::vector< cv::Vec4i > & hierarchy, std::vector<std::pair<float, int> >& quads, int class_id);

inline bool less_predCustom(const std::pair<float, int>& p1, const std::pair<float, int>& p2);

static void countClassesCustom(const std::vector<std::pair<float, int> >& pairs, size_t idx1, size_t idx2, std::vector<int>& counts);

static void fillQuadsCustom(Mat & white, Mat & black, double white_thresh, double black_thresh, std::vector<std::pair<float, int> > & quads);

static bool checkQuadsCustom(std::vector<std::pair<float, int> > & quads, const cv::Size & size);

int checkChessboardBinaryCustom(const cv::Mat & img, const cv::Size & size);

bool findChessboardCornersCustom(InputArray image_, Size pattern_size, OutputArray corners_, int flags);

#endif //__CUSTOM_H__s