/*
 * main.cpp
 * Point d'entrée principal de l'application UltrafastLaneDetector pour les videos.
 * Initialise les composants, charge le modèle et exécute le pipeline de détection de lignes.
 * Auteur : Oumar Koné
 * Licence : MIT
 */

#include "UltrafastLaneDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const std::string model_path = "../models/model_integer_quant.tflite";
    const std::string video_path = "../video.mp4";
    
    try {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) throw std::runtime_error("Cannot open video");
        
        UltrafastLaneDetector detector(model_path, ModelType::TUSIMPLE);
        cv::namedWindow("Detected lanes", cv::WINDOW_NORMAL);

        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            
            cv::Mat output = detector.detectLanes(frame);
            cv::imshow("Detected lanes", output);
            
            if (cv::waitKey(1) == 'q') break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}