#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "custom.cpp"

#include <stdio.h>
#include <iostream>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
// Both width and height of the pattern should have bigger than 2 in function 'findChessboardCorners'
//  Defining the dimensions of checkerboard
// int CHECKERBOARD[2]{3,24};
int CHECKERBOARD[2]{1, 24};

//Permet d'afficher des images sur des fenêtres différentes
void afficherImage(Mat image, const std::string& name){
  namedWindow(name, WINDOW_NORMAL);
  cv::imshow(name, image);
  cv::resizeWindow(name, 600, 600);
}

/**
 * @brief Calcul la longueur en pixel des carrés d'un damier sur une image.
 * 
 * @param fileName Nom du fichier a analyser.
 * @param pattern Taille de la mire/damier.
 * @return La longueur en pixel des carrés si trouvée, -1 sinon.
 */
double calculeEchelleDamier(const std::string& fileName, const int pattern[2], bool debug = false){
  //---------------------
  //Constantes
  const int min_dilations = 0;
  const int max_dilations = 5;
  //Variables
  bool found = false;
  double pixelWidth = -1;
  std::vector<cv::Point2f> out_corners;
  int prev_sqr_size = 0;
  ChessBoardDetector detector(cv::Size(pattern[0], pattern[1]));
  //---------------------

  if (debug)
  {
    std::cout << "Pattern size: width:" << detector.pattern_size.width << " heigh: " << detector.pattern_size.height << endl; 
  }
  

  //On charge l'image
  Mat image = imread(fileName, cv::IMREAD_COLOR);

  if(debug){
    afficherImage(image, "Originale");
  }

  // On convertie l'image en niveau de gris.
  Mat gray;
  cvtColor(image, gray, COLOR_BGR2GRAY);

  //On binarise l'image.
  Mat binary = gray.clone();
  icvBinarizationHistogramBased(binary);

  if(debug){
    afficherImage(binary, "Binarisation");
  }

  //On teste plusieurs niveaux de dilatation.
  for (size_t i = min_dilations; i < max_dilations; i++)
  {
    dilate(binary, binary, Mat(), Point(-1, -1), 1);
    if(debug){
      afficherImage(binary, "Dilatation");
    }

    detector.generateQuadsCustom(binary, 0);

    bool found = detector.processQuadsCustom2(out_corners, prev_sqr_size, binary, fileName, &pixelWidth);


    //Si on a trouvé un pattern qui corresponds, on arrête les itérationsé
    if (found)
    {
      int k = 3 + 2 * (i);
      pixelWidth = pixelWidth + k - 1;
      break;
    }
  }
  return pixelWidth;
}

int main(int argc, const char **argv)
{
  double pixelWidth = -1;
  int nbFichierTraite = 0;
  //calculeEchelleDamier("../Data/OLD/IMG4L-1.jpg",CHECKERBOARD,false);
  const std::string dossier = "../Data/OLD/";
  for (const auto& fichier : std::filesystem::directory_iterator(dossier)){
    if(fichier.path().extension() == ".jpg"){
      std::cout << "------------" << endl << "FICHIER: " << fichier.path().string() << endl;
      pixelWidth = calculeEchelleDamier(fichier.path().string(),CHECKERBOARD,false);
      if(pixelWidth == -1){
        std::cout << "Pattern non trouvé."<< endl;
      } else {
        std::cout << "Pattern trouvé. Longueur: " << pixelWidth << endl;
      }
      nbFichierTraite++;
      std::cout << "------------" << endl << endl;;
    }
  }
  std::cout << "Nombre de fichier traite(s): " << nbFichierTraite << endl;
  waitKey(0);
  return 0;
}