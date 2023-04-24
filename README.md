# M1-CalibrationDamier
Dans le cadre d’une collaboration avec l’INRAE sur un projet de recherche lié au traitement et l’analyse d’images de bois, le but du projet de recherche est d’évaluer de la qualité du bois coupé à partir d’images prises à l’aide d’appareils photos ou de caméras situées à la scierie ou en bord de route.

# Versions & Library nécessaires
### Programme principal
OpenCV, version 4.2+
Compilation: C++ 17 or later

### Script pour les graphiques
Python3+

# Compilation
```
cd CodeSample
cmake .
make
```

# Execution
Usage: 
```
./main [INPUT FOLDER] [-d] [-o OUTPUT FOLDER]
```
## Arguments
INPUT FOLDER   : Obligatoire, le chemin du répertoire d'entrée
-d             : Optionnel, active le mode de débogage
-o OUTPUT FOLDER : Optionnel, spécifie le chemin du répertoire de sortie (nécessite -o)

## Fonctionnement
Si -o n'est pas spécifié, les fichiers de sorties se situeront dans CodeSample/Data/Results.

La fonction "bool isSquare(int l1, int l2, int l3, int l4)" dans le fichier custom.cpp contient une variables de tolerance qui permet d'éliminer les quads qui "ne sont pas assez carrés"; il s'agit d'un pourcentage. Si les longueurs sont trop éloignées les unes des autres a ce pourcentage près, alors le quad n'est pas pris en compte. 
Cette fonction permet d'éliminer les images avec des mires trop penchés qui a l'heure actuelle posent problème.