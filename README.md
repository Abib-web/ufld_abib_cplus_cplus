# 🚗 Ultrafast Lane Detection – Embedded C++ Project

Ce projet implémente un système de détection de lignes routières en temps réel à l’aide de **TensorFlow Lite** et **OpenCV**, conçu pour les plateformes embarquées.

## 📂 Structure du projet

- `src/` : Code source C++ (prétraitement, inférence, post-traitement)
- `include/` : Headers du projet
- `models/` : Modèle `.tflite` (non inclus – voir ci-dessous)
- `input_images/` : Images de test (non inclus – voir ci-dessous)
- `main.cpp` : Exemple d'utilisation
- `CMakeLists.txt` : Fichier de configuration pour compilation

## ⚙️ Dépendances

- g++ version 9.4.0
- flatc version 23.5.26
- OpenCV 4.2.0
- TensorFlow Lite v2.14.0
- bazel 6.1.0
- CMake (version ≥ 3.10)

## 🧠 Modèle

Le modèle utilisé est un réseau **UFLD** (Ultra Fast Lane Detection), exporté en `.tflite`.  
Pour des raisons de taille/licence, le fichier n’est pas inclus dans ce dépôt.

### 📥 Télécharger le modèle :
> [Lien Google Drive / HuggingFace / Autre à insérer ici]

Placez le fichier dans le dossier :

## models/ultrafast_lane.tflite


## 🖼️ Images de test

Les images de test sont également exclues du dépôt.  
Tu peux en télécharger depuis le dataset [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues) ou utiliser tes propres images.

### 📥 Exemple de lien :
> [Lien vers images ou dataset compressé]

Placez-les dans le dossier : input_mages


## 🚀 Exécution

```bash
mkdir build && cd build
cmake ..
make
./ufld_image
