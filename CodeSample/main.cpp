#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "custom.cpp"
#include <chrono>

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

struct ImageData
{
  std::string FileName;
  bool found;
  double longueurTotal;
  double longueurGarde;
  int nbCarreTotal;
  int nbCarreGarde;
};

void
ecrireCSV(const std::vector<ImageData> &donnees, const std::string &nomFichier)
{
  std::string dossierResultats = "../Data/Results";
  std::filesystem::create_directories(dossierResultats);      // création du dossier s'il n'existe pas déjà
  std::ofstream fichier(dossierResultats + "/" + nomFichier); // chemin complet du fichier
  if (fichier.is_open())
  {
    fichier << "NomFichier;Trouvé;Longueur;NbCarre\n"; // Écriture de l'en-tête CSV
    for (const auto &donnee : donnees)
    {
      fichier << donnee.FileName << ";" << (donnee.found ? "oui" : "non") << ";" << donnee.longueurGarde << ";" << donnee.nbCarreTotal << "\n";
    }
    fichier.close(); // Fermeture du fichier
    std::cout << "Les données ont été écrites dans le fichier " << nomFichier << std::endl;
  }
  else
  {
    std::cerr << "Impossible d'ouvrir le fichier " << nomFichier << " pour écriture." << std::endl;
  }
}

// Permet d'afficher des images sur des fenêtres différentes
void afficherImage(Mat image, const std::string &name)
{
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
double calculeEchelleDamier(const std::string &fileName, const int pattern[2], bool debug = false)
{
  //---------------------
  // Constantes
  const int min_dilations = 0;
  const int max_dilations = 3;
  // Variables
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

  // On charge l'image
  Mat image = imread(fileName, cv::IMREAD_COLOR);

  if (debug)
  {
    afficherImage(image, "Originale");
  }

  // On convertie l'image en niveau de gris.
  Mat gray;
  cvtColor(image, gray, COLOR_BGR2GRAY);

  // On binarise l'image.
  Mat binary = gray.clone();
  int t = 140;

  threshold(gray, binary, t, 255, 0);

  if (debug)
  {
    afficherImage(binary, "Binarisation");
  }
  // On teste plusieurs niveaux de dilatation.
  for (size_t i = min_dilations; i < max_dilations; i++)
  {
    std::cout << "Dilatation = " << i << endl;
    dilate(binary, binary, Mat(), Point(-1, -1), 1);
    if (debug)
    {
      afficherImage(binary, "Dilatation");
    }

    detector.generateQuadsCustom(binary, 0);

    bool found = detector.processQuadsCustom2(out_corners, prev_sqr_size, binary, fileName, &pixelWidth);

    // Si on a trouvé un pattern qui corresponds, on arrête les itérationsé
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
  bool found = 0;
  double pixelWidth = -1;
  int nbFichierTraite = 0;
  int nbPatternTrouve = 0;

  std::vector<ImageData> datas;

  // calculeEchelleDamier("../Data/OLD/D01-L4-BBF-4.jpg", CHECKERBOARD, true);

  const std::string dossier = "../Data/OLD/";
  auto start0 = std::chrono::high_resolution_clock::now();
  for (const auto &fichier : std::filesystem::directory_iterator(dossier))
  {
    if (fichier.path().extension() == ".jpg" || fichier.path().extension() == ".JPG")
    {
      found = 0;
      std::cout << "------------" << endl
                << "FICHIER: " << fichier.path().string() << endl;
      auto start = std::chrono::high_resolution_clock::now();
      pixelWidth = calculeEchelleDamier(fichier.path().string(), CHECKERBOARD, false);
      auto end = std::chrono::high_resolution_clock::now();
      if (pixelWidth == -1)
      {
        std::cout << "Pattern non trouvé." << endl;
      }
      else
      {
        std::cout << "Pattern trouvé. Longueur: " << pixelWidth << endl;
        found = true;
        nbPatternTrouve++;
      }
      nbFichierTraite++;
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Temps d'execution: " << elapsed.count() << "sec." << endl;
      std::cout << "------------" << endl
                << endl;
    }
    ImageData data = {fichier.path().string(),found,0, pixelWidth,0,0};
    datas.push_back(data);
  }
  auto end0 = std::chrono::high_resolution_clock::now();
  std::cout << "Nombre de fichier traite(s): " << nbFichierTraite << endl;
  std::cout << "Nombre de fichi: " << datas.size() << endl;
  
  std::cout << "Nombre de pattern trouve(s): " << nbPatternTrouve << endl;
  std::chrono::duration<double> elapsed0 = end0 - start0;
  std::cout << "Temps d'execution total: " << elapsed0.count() << "sec." << endl;

  std::string nomFichier = "Resultats.csv";
  ecrireCSV(datas, nomFichier);

  waitKey(0);
  return 0;
}