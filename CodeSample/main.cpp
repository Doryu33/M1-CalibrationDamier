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
#include <libgen.h>

using namespace cv;
using namespace std;
// Both width and height of the pattern should have bigger than 2 in function 'findChessboardCorners'
// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{1, 24};
// Nombre minimum de quad a trouver.
int MINQUAD = 10;
bool DEBUG_MOD = false;
std::string OUTPUT_FOLDER_PATH = "../Data/Results";

void afficherImageData(const ImageData &imageData)
{
  std::cout << "FileName: " << imageData.FileName << std::endl;
  std::cout << "mireTrouvee: " << imageData.mireTrouvee << std::endl;
  std::cout << "nbCarresMire: " << imageData.nbCarresMire << std::endl;
  std::cout << "nbCarresDetectes: " << imageData.nbCarresDetectes << std::endl;
  std::cout << "moyenneLongueurCote_Pixels: " << imageData.moyenneLongueurCote_Pixels << std::endl;
  std::cout << "medianeLongueurCote_Pixels: " << imageData.medianeLongueurCote_Pixels << std::endl;
  std::cout << "minLongueurCote_Pixels: " << imageData.minLongueurCote_Pixels << std::endl;
  std::cout << "maxLongueurCote_Pixels: " << imageData.maxLongueurCote_Pixels << std::endl;
  std::cout << "ecartTypeLongueurCote_Pixels: " << imageData.ecartTypeLongueurCote_Pixels << std::endl;
  std::cout << "nombreCarresDetectesSansExtremes: " << imageData.nombreCarresDetectesSansExtremes << std::endl;
  std::cout << "moyenneLongueurCoteSansExtremes_Pixels: " << imageData.moyenneLongueurCoteSansExtremes_Pixels << std::endl;
  std::cout << "medianeLongueurCoteSansExtremes_Pixels: " << imageData.medianeLongueurCoteSansExtremes_Pixels << std::endl;
  std::cout << "minLongueurCoteSansExtremes_Pixels: " << imageData.minLongueurCoteSansExtremes_Pixels << std::endl;
  std::cout << "maxLongueurCoteSansExtremes_Pixels: " << imageData.maxLongueurCoteSansExtremes_Pixels << std::endl;
  std::cout << "ecartTypeLongueurCoteSansExtremes_Pixels: " << imageData.ecartTypeLongueurCoteSansExtremes_Pixels << std::endl;
}

/**
 * @brief Permet de créer un fichier .CSV avec les mesures realisees par le programme
 *
 * @param donnees Vecteur de structure "ImageData".
 * @param nomFichier nom de sortie du fichier.
 */
void ecrireCSV(const std::vector<ImageData> &donnees, const std::string &nomFichier)
{
  std::string dossierResultats = OUTPUT_FOLDER_PATH;
  std::filesystem::create_directories(dossierResultats);      // création du dossier s'il n'existe pas déjà
  std::ofstream fichier(dossierResultats + "/" + nomFichier); // chemin complet du fichier
  if (fichier.is_open())
  {
    // Écriture de l'en-tête CSV
    fichier << "nomImage;mireTrouvee;nombreCarresMire;nombreCarresDetectes;moyenneLongueurCote_Pixels;medianeLongueurCote_Pixels;minLongueurCote_Pixels;maxLongueurCote_Pixels;ecartTypeLongueurCote_Pixels;nombreCarresDetectesSansExtremes;moyenneLongueurCoteSansExtremes_Pixels;medianeLongueurCoteSansExtremes_Pixels;minLongueurCoteSansExtremes_Pixels;maxLongueurCoteSansExtremes_Pixels;ecartTypeLongueurCoteSansExtremes_Pixels \n";
    for (const auto &donnee : donnees)
    {
      fichier << donnee.FileName << ";" << (donnee.mireTrouvee ? "Oui" : "Non")
              << ";" << donnee.nbCarresMire << ";" << donnee.nbCarresDetectes
              << ";" << donnee.moyenneLongueurCote_Pixels
              << ";" << donnee.medianeLongueurCote_Pixels
              << ";" << donnee.minLongueurCote_Pixels
              << ";" << donnee.maxLongueurCote_Pixels
              << ";" << donnee.ecartTypeLongueurCote_Pixels
              << ";" << donnee.nombreCarresDetectesSansExtremes
              << ";" << donnee.moyenneLongueurCoteSansExtremes_Pixels
              << ";" << donnee.medianeLongueurCoteSansExtremes_Pixels
              << ";" << donnee.minLongueurCoteSansExtremes_Pixels
              << ";" << donnee.maxLongueurCoteSansExtremes_Pixels
              << ";" << donnee.ecartTypeLongueurCoteSansExtremes_Pixels << "\n";
    }
    fichier.close(); // Fermeture du fichier
    std::cout << "Les données ont été écrites dans le fichier " << nomFichier << std::endl;
  }
  else
  {
    std::cerr << "Impossible d'ouvrir le fichier " << nomFichier << " pour écriture." << std::endl;
  }
}

/**
 * @brief Permet de créer un fichier CSV contenant les coordonnées de chaque coins de la mire
 * detectée sur l'image.
 * @param nomFichier
 * @param quads
 */
void img2CSV(const std::string &nomFichier, const std::vector<QuadData> &quads)
{
  // Extraire le nom de fichier sans le chemin ni l'extension
  char *nomFichierSansChemin = basename(const_cast<char *>(nomFichier.c_str()));
  std::string nomFichierSansExtension = std::string(nomFichierSansChemin).substr(0, std::string(nomFichierSansChemin).find_last_of("."));

  // Créer le chemin dossier
  std::string cheminDossier = OUTPUT_FOLDER_PATH;
  std::filesystem::create_directories(cheminDossier);

  // Créer le chemin complet du fichier en concaténant le nom de fichier et le chemin vers le dossier
  std::string cheminFichier = cheminDossier + nomFichierSansExtension + ".csv";

  // Ouvrir le fichier en mode écriture
  std::ofstream fichier(cheminFichier);

  // Vérifier si le fichier a été ouvert correctement
  if (!fichier.is_open())
  {
    std::cerr << "Impossible d'ouvrir le fichier " << cheminFichier << std::endl;
    return;
  }

  // Écrire l'en-tête du fichier
  fichier << "idQuad;hg.x;hg.y;hd.x;hd.y;bd.x;bd.y;bg.x;bg.y\n";

  // Écrire les données de chaque QuadData dans le fichier
  for (const auto &quad : quads)
  {
    fichier << quad.idQuad << ";"
            << quad.hg.x << ";" << quad.hg.y << ";"
            << quad.hd.x << ";" << quad.hd.y << ";"
            << quad.bd.x << ";" << quad.bd.y << ";"
            << quad.bg.x << ";" << quad.bg.y << "\n";
  }

  // Fermer le fichier
  fichier.close();
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
ImageData calculeEchelleDamier(const std::string &fileName, const int pattern[2], bool debug)
{
  //---------------------
  // Constantes
  const int min_dilations = 0;
  const int max_dilations = 3;

  // Variables
  bool found = false;
  ImageData data = {fileName, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<QuadData> imageQuadsData;
  double pixelWidth = -1;
  std::vector<cv::Point2f> out_corners;
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
  int t = 80;

  while (t < 170 && !found)
  {
    imageQuadsData.clear();
    std::cout << "FoundW = " << found << endl;
    threshold(gray, binary, t, 255, 0);
    std::cout << "Threshold = " << t << endl;

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

      if (debug)
      {
        afficherImage(binary, "GenerateQuad");
      }

      bool found = detector.findTarget(out_corners, binary, fileName, &data, &imageQuadsData, i, MINQUAD, debug);

      // Si on a trouvé un pattern qui corresponds, on arrête les itérations
      if (found)
      {
        std::cout << "Found?" << found << endl;
        break;
      }
    }
    t = t + 45;
  }

  // Ne prends pas en compte les dilatations.
  // Les coordonnées sont donc un peu faussée.
  img2CSV(data.FileName, imageQuadsData);
  return data;
}

int main(int argc, const char **argv)
{
  std::string INPUT_FOLDER_PATH;
  // Variables
  bool found = 0;
  double pixelWidth = -1;
  int nbFichierTraite = 0;
  int nbPatternTrouve = 0;

  // Verification des arguments du programme
  if (argc < 1)
  {
    std::cerr << "Usage: " << argv[0] << " [directory path]" << std::endl;
    std::cerr << "\nPlease refer to the Read Me." << std::endl;
    return 1;
  }

  INPUT_FOLDER_PATH = argv[1];
  if (!std::filesystem::is_directory(INPUT_FOLDER_PATH))
  {
    std::cout << "Erreur. Le chemin \"" << INPUT_FOLDER_PATH << "\" n'est pas un repertoir valide." << std::endl;
    exit(1);
  }

  for (int i = 2; i < argc; i++)
  {
    // Modifie le mode DEBUG
    if (strcmp(argv[i], "-d") == 0)
    {
      DEBUG_MOD = true;
    }
    // Modifie le repertoir de sortie
    else if (strcmp(argv[i], "-o") == 0)
    {
      i++;
      if (i < argc)
      {
        OUTPUT_FOLDER_PATH = argv[i];
      }
      else
      {
        std::cout << "Il manque un argument: [OUTPUT_FOLDER_PATH] n'est pas renseigné après le -o." << std::endl;
        exit(1);
      }
    }
  }

  //==========================================

  // Redirection de la sortie standard dans un fichier logs.txt.
  // Créer un nom de fichier unique à partir de l'heure actuelle
  if (false)
  {
    std::string logs_folder_name = "LOGS";
    std::filesystem::create_directories(logs_folder_name);

    std::time_t t = std::time(nullptr);
    char timestamp[30];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&t));
    std::string filename = std::string(timestamp) + "_logs.txt";

    // Créer le fichier de log
    std::ofstream log_file(logs_folder_name + "/" + filename); // Utiliser "/" comme délimiteur de répertoire sur Unix/Linux

    // Rediriger la sortie standard vers le fichier de log
    std::streambuf *old_cout_buffer = std::cout.rdbuf();
    std::cout.rdbuf(log_file.rdbuf());
  }
  //===========================================================

  //===========================================================
  // Instructions principales:
  // Vecteur qui va contenir les datas des images (voir structure plus haut)
  std::vector<ImageData> datas;

  // Repertoire qui contient les images
  const std::string dossier = INPUT_FOLDER_PATH;
  // Pour avoir un timer
  auto start0 = std::chrono::high_resolution_clock::now();
  for (const auto &fichier : std::filesystem::directory_iterator(dossier))
  {
    ImageData data = {};
    if (fichier.path().extension() == ".jpg" || fichier.path().extension() == ".JPG" || fichier.path().extension() == ".jpeg")
    {
      found = 0;
      std::cout << "------------" << endl
                << "FICHIER: " << fichier.path().string() << endl;
      auto start = std::chrono::high_resolution_clock::now();
      data = calculeEchelleDamier(fichier.path().string(), CHECKERBOARD, DEBUG_MOD);
      auto end = std::chrono::high_resolution_clock::now();
      if (!data.mireTrouvee)
      {
        std::cout << "Pattern non trouvé." << endl;
      }
      else
      {
        std::cout << "Pattern trouvé. Longueur: " << data.moyenneLongueurCoteSansExtremes_Pixels << endl;
        found = true;
        nbPatternTrouve++;
      }
      nbFichierTraite++;
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Temps d'execution: " << elapsed.count() << "sec." << endl;
      std::cout << "------------" << endl
                << endl;
    }
    datas.push_back(data);
  }
  auto end0 = std::chrono::high_resolution_clock::now();
  std::cout << "Nombre de fichier du repertoire: " << datas.size() << endl;
  std::cout << "Nombre de fichier traite(s): " << nbFichierTraite << endl;
  std::cout << "Nombre de pattern trouve(s): " << nbPatternTrouve << endl;
  std::chrono::duration<double> elapsed0 = end0 - start0;
  std::cout << "Temps d'execution total: " << elapsed0.count() << "sec." << endl;
  std::string nomFichier = "Resultats.csv";
  ecrireCSV(datas, nomFichier);

  waitKey(0);
  // Restaurer la sortie standard
  //std::cout.rdbuf(old_cout_buffer);

  return 0;
}