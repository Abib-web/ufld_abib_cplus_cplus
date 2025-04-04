/*
 * main_images.cpp
 * Point d'entrée principal de l'application UltrafastLaneDetector pour les images.
 * Initialise les composants, charge le modèle et exécute le pipeline de détection de lignes.
 * Auteur : Oumar Koné
 * Licence : MIT
 */

#include "UltrafastLaneDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>

bool directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

int main() {
    const std::string model_path = "../models/model_integer_quant.tflite";
    const std::string input_folder = "../input_images";
    
    try {
        UltrafastLaneDetector detector(model_path, ModelType::TUSIMPLE);
        DIR *dir = opendir(input_folder.c_str());
        
        if (dir) {
            while (struct dirent *ent = readdir(dir)) {
                std::string filename = ent->d_name;
                if (filename == "." || filename == "..") continue;
                
                std::string image_path = input_folder + "/" + filename;
                cv::Mat img = cv::imread(image_path);
                
                if (!img.empty()) {
                    cv::Mat output = detector.detectLanes(img);
                    
                    // Affiche l'image au lieu de l'enregistrer
                    cv::imshow("Détection de voies", output);
                    std::cout << "Appuyez sur une touche pour continuer..." << std::endl;
                    cv::waitKey(0);  // Attend une touche
                }
            }
            closedir(dir);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}