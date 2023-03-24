#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include "custom.cpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;
// Both width and height of the pattern should have bigger than 2 in function 'findChessboardCorners'
//  Defining the dimensions of checkerboard
// int CHECKERBOARD[2]{3,24};
int CHECKERBOARD[2]{1, 24};

#include <opencv2/opencv.hpp>

int main(int argc, const char **argv)
{
  // Charger l'image
  Mat image = imread(argv[1], 1);
  // Convertir l'image en niveaux de gris
  Mat gray;
  cvtColor(image, gray, COLOR_BGR2GRAY);

  // Seuiller l'image pour isoler le damier
  Mat binary = gray.clone();
  // threshold(gray, binary, 150, 255, THRESH_BINARY_INV);
  icvBinarizationHistogramBased(binary);

  std::vector<cv::Point2f> out_corners;
  int prev_sqr_size = 0;

  ChessBoardDetector detector(cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]));

  
  std::cout << "Pattern size: width:" << detector.pattern_size.width << " heigh: " << detector.pattern_size.height << endl; 

  dilate(binary, binary, Mat(), Point(-1, -1), 1);
  namedWindow("Image: After dilation", WINDOW_NORMAL);
  cv::imshow("Image: After dilation", binary);
  cv::resizeWindow("Image: After dilation", 600, 600);

  detector.generateQuadsCustom(binary, 0);
  bool test = detector.processQuadsCustom(out_corners, prev_sqr_size, binary);
  std::cout << test << endl;
  if(test)
  {
    //------------------------
    // Creation d'une nouvelle image pour pouvoir dessiner le rectangle en couleur
    Mat img2;
    cv::cvtColor(binary, img2, 8);
    // On dessine un rectangle pour delimiter la zone du patern trouve
    rectangle(img2, out_corners[0], out_corners[out_corners.size() - 1], Scalar(0, 0, 255), 8, LINE_8);
    namedWindow("Image: After rectangle", WINDOW_NORMAL);
    cv::imshow("Image: After rectangle", img2);
    cv::resizeWindow("Image: After rectangle", 600, 600);
    //------------------------
  }

  // Afficher l'image résultante
  namedWindow("Result", WINDOW_NORMAL);
  imshow("Result", image);
  cv::resizeWindow("Result", 600, 600);

  waitKey(0);

  return 0;
}

/*
int main(int argc, const char **argv)
{

  int key = 0;
  if(argc==3) {
    CHECKERBOARD[0] = atoi(argv[1]);
    CHECKERBOARD[1] = atoi(argv[2]);
  }
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++) {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j,i,0));
  }
  //Defining images
  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners
  std::vector<cv::Point2f> corner_pts;
  bool success;

  VideoCapture capture;
  capture.open(0);
  if (!capture.isOpened()){
    std::cout << "Pas de camera. Ouverture de l'image: " << argv[1] << endl << endl;
    frame = imread( argv[1], 1 );
    if ( !frame.data )
    {
        printf("No image data \n");
        return -1;
    }
    //Detect chessboard inner corner
      cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
      // Finding checker board corners
      // If desired number of corners are found in the image then success = true
      success = findChessboardCornersCustom(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

      // If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
      if(success) {
        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.001);

        // refining pixel coordinates for given 2d points.
        cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
        //Display the result
      }
      flip(frame, frame, 1);
      namedWindow("Image", WINDOW_NORMAL);
      cv::imshow("Image",frame);
      cv::resizeWindow("Image", 600, 600);
    waitKey(0);
    return 0;
  }
  else {
    capture.read(frame);
    while(key!='q') {
      capture.read(frame);
      //Detect chessboard inner corner
      cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
      // Finding checker board corners
      // If desired number of corners are found in the image then success = true
      success = findChessboardCornersCustom(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

      // If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
      if(success) {
        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.001);

        // refining pixel coordinates for given 2d points.
        cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

        //objpoints.push_back(objp);
        //imgpoints.push_back(corner_pts);
        //Display the result
      }
      flip(frame, frame, 1);
      cv::imshow("Image",frame);
      key = waitKey(1);
    }
  }


  cv::destroyAllWindows();

  return 0;
}
*/