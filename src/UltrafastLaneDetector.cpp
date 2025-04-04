/*
 * UltrafastLaneDetector.cpp
 * Author: Oumar Koné
 * Description: Lane detection using TensorFlow Lite and OpenCV.
 *              Processes images and extracts lane lines using a trained model.
 * License: MIT
 */

#include "UltrafastLaneDetector.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include <fstream>
#include <stdexcept>
#include <cfloat>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace std;

ModelConfig::ModelConfig(ModelType model_type) {
    if (model_type == ModelType::TUSIMPLE) {
        img_w = 1280;
        img_h = 720;
        row_anchor = {64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284};
        griding_num = 100;
        cls_num_per_lane = 56;
    } else {
        img_w = 1640;
        img_h = 590;
        row_anchor = {121,131,141,150,160,170,180,189,199,209,219,228,238,248,258,267,277,287};
        griding_num = 200;
        cls_num_per_lane = 18;
    }
}

UltrafastLaneDetector::UltrafastLaneDetector(const string& model_path, ModelType model_type) : cfg(model_type) {
    initializeModel(model_path);
}

UltrafastLaneDetector::~UltrafastLaneDetector() {}

void UltrafastLaneDetector::initializeModel(const string& model_path) {
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        throw runtime_error("Failed to load TFLite model");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk) {
        throw runtime_error("Failed to initialize interpreter");
    }

    const auto* input_tensor = interpreter->input_tensor(0);
    input_height = input_tensor->dims->data[1];
    input_width = input_tensor->dims->data[2];
    channels = input_tensor->dims->data[3];
}

cv::Mat UltrafastLaneDetector::detectLanes(const cv::Mat& image, bool draw_points) {
    auto input_tensor = prepareInput(image);
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, input_tensor.data(), input_tensor.size() * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) {
        throw runtime_error("Inference failed");
    }

    float* output = interpreter->typed_output_tensor<float>(0);
    TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    auto result = processOutput(output);
    return drawLanes(image, result.first, result.second, draw_points);
}

vector<float> UltrafastLaneDetector::prepareInput(const cv::Mat& image) {
    cv::Mat img_rgb;
    cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(input_width, input_height));

    vector<float> input_data(input_width * input_height * channels);
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};

    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float pixel = img_resized.at<cv::Vec3b>(y, x)[c] / 255.0f;
                input_data[(y * input_width + x) * channels + c] = (pixel - mean[c]) / std[c];
            }
        }
    }

    return input_data;
}

template<typename T>
std::vector<T> UltrafastLaneDetector::arange(T start, T stop, T step) {
    std::vector<T> values;
    if (step == 0) return values;
    
    // Calcul plus précis de la taille
    size_t size = static_cast<size_t>(std::round((stop - start) / step)) + 1;
    if (size > values.max_size()) {
        throw std::overflow_error("Requested size exceeds vector max_size");
    }
    values.reserve(size);
    
    for (T value = start; value <= stop; value += step) {
        values.push_back(value);
        if (values.size() == values.capacity()) break; 
    }
    return values;
}

template<typename T>
UltrafastLaneDetector::Tensor3D<T>::Tensor3D(size_t dim1, size_t dim2, size_t dim3) 
    : dims{dim1, dim2, dim3}, data(dim1 * dim2 * dim3) {}

template<typename T>
T& UltrafastLaneDetector::Tensor3D<T>::operator()(size_t i, size_t j, size_t k) {
    return data[i * dims[1] * dims[2] + j * dims[2] + k];
}

template<typename T>
void UltrafastLaneDetector::Tensor3D<T>::reshape(size_t new_dim1, size_t new_dim2, size_t new_dim3) {
    if (new_dim1 * new_dim2 * new_dim3 != data.size()) {
        throw std::invalid_argument("New dimensions must match total size");
    }
    dims = {new_dim1, new_dim2, new_dim3};
}

std::vector<std::vector<std::vector<float>>> 
UltrafastLaneDetector::reshapeTensor3D(const std::vector<float>& data,
                                     size_t dim1, size_t dim2, size_t dim3) {
    if (data.size() != dim1 * dim2 * dim3) {
        throw std::invalid_argument("Dimensions do not match data size");
    }

    std::vector<std::vector<std::vector<float>>> result(
        dim1, std::vector<std::vector<float>>(
            dim2, std::vector<float>(dim3)));

    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                result[i][j][k] = data[i * dim2 * dim3 + j * dim3 + k];
            }
        }
    }

    return result;
}


std::vector<std::vector<int>> UltrafastLaneDetector::argmax_axis0_3d(
    const std::vector<std::vector<std::vector<float>>>& processed_output) 
{
    if (processed_output.empty() || processed_output[0].empty() || processed_output[0][0].empty()) {
        return {};
    }

    size_t depth = processed_output.size();
    size_t rows = processed_output[0].size();
    size_t cols = processed_output[0][0].size();
    std::vector<std::vector<int>> result(rows, std::vector<int>(cols));

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            float max_val = processed_output[0][row][col];
            int max_idx = 0;

            for (size_t d = 1; d < depth; ++d) {
                if (processed_output[d][row][col] > max_val) {
                    max_val = processed_output[d][row][col];
                    max_idx = d;
                }
            }

            result[row][col] = max_idx;
        }
    }

    return result;
}
std::vector<std::vector<std::vector<float>>> UltrafastLaneDetector::softmax_axis0(
    const std::vector<std::vector<std::vector<float>>>& input) 
{
    if (input.empty() || input[0].empty() || input[0][0].empty()) {
        throw std::invalid_argument("Input tensor cannot be empty");
    }

    const size_t depth = input.size() - 1;
    const size_t rows = input[0].size();
    const size_t cols = input[0][0].size();
    
    std::vector<std::vector<std::vector<float>>> output(
        depth,
        std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    for (size_t l = 0; l < rows; ++l) {
        for (size_t p = 0; p < cols; ++p) {
            float max_val = -FLT_MAX;
            for (size_t g = 0; g < depth; ++g) {
                if (input[g][l][p] > max_val) {
                    max_val = input[g][l][p];
                }
            }
            
            float sum_exp = 0.0f;
            std::vector<float> exps(depth);
            for (size_t g = 0; g < depth; ++g) {
                exps[g] = exp(input[g][l][p] - max_val);
                sum_exp += exps[g];
            }
            
            for (size_t g = 0; g < depth; ++g) {
                output[g][l][p] = exps[g] / sum_exp;
            }
        }
    }
    
    return output;
}

std::vector<std::vector<std::vector<float>>> UltrafastLaneDetector::reverse_second_dim(
    const float* output, 
    int griding_num, 
    int num_lanes, 
    int points_per_lane) 
{
    std::vector<std::vector<std::vector<float>>> processed_output(
        griding_num,
        std::vector<std::vector<float>>(num_lanes, std::vector<float>(points_per_lane)));

    for (int g = 0; g < griding_num; ++g) {
        for (int l = 0; l < num_lanes; ++l) {
            for (int p = 0; p < points_per_lane; ++p) {
                int idx = g * num_lanes * points_per_lane + l * points_per_lane + p;
                processed_output[g][l][p] = output[idx];
            }
        }
    }

    for (auto& grid : processed_output) {
        std::reverse(grid.begin(), grid.end());
    }

    return processed_output;
}
pair<vector<vector<cv::Point>>, vector<bool>> UltrafastLaneDetector::processOutput(const float* output) {
    vector<vector<cv::Point>> lane_points_mat;
    vector<bool> lanes_detected(4, false);

    // Get tensor dimensions
    TfLiteTensor* tensor = interpreter->output_tensor(0);
    const int actual_grid = tensor->dims->data[1];
    const int actual_lanes = tensor->dims->data[2];
    const int actual_points = tensor->dims->data[3];
    // 1. Redimensionner correctement le tenseur de sortie (grid,lanes,points)
    vector<vector<vector<float>>> reshaped_output(
        actual_grid, 
        vector<vector<float>>(actual_lanes, 
        vector<float>(actual_points)));

    for (int g = 0; g < actual_grid; ++g) {
        for (int l = 0; l < actual_lanes; ++l) {
            for (int p = 0; p < actual_points; ++p) {
                reshaped_output[g][l][p] = output[g * actual_lanes * actual_points + l * actual_points + p];
            }
        }
    }

    // 2. Inverser la deuxième dimension comme en Python
    for (auto& grid : reshaped_output) {
        reverse(grid.begin(), grid.end());
    }

    // 3. Calculer les indices comme en Python (1..griding_num)
    vector<float> idx(cfg.griding_num);
    for (int i = 0; i < cfg.griding_num; ++i) {
        idx[i] = i + 1;
    }

    // 4. Calculer softmax probabilities
    auto prob = softmax_axis0(reshaped_output);
    // 5. Calculer loc = sum(prob * idx)
    vector<vector<float>> loc(actual_lanes, vector<float>(actual_points, 0.0f));
    for (int g = 0; g < actual_grid-1; ++g) {
        for (int l = 0; l < actual_lanes; ++l) {
            for (int p = 0; p < actual_points; ++p) {
                loc[l][p] += prob[g][l][p] * idx[g];
            }
        }
    }
    // 6. Calculer argmax
    auto max_indices = argmax_axis0_3d(reshaped_output);

    // 7. Appliquer la condition comme en Python
    for (int l = 0; l < actual_lanes; ++l) {
        for (int p = 0; p < actual_points; ++p) {
            if (max_indices[l][p] == cfg.griding_num) {
                loc[l][p] = 0;
            }
        }
    }
    // Création de col_sample
    vector<float> col_sample(cfg.griding_num);
    float col_sample_w;
    for(int i = 0; i < cfg.griding_num; ++i) {
        col_sample[i] = i * (800.0f) / (cfg.griding_num - 1);  // Retirer le -1 dans (800.0f - 1)
    }
    col_sample_w = col_sample[1] - col_sample[0];
    // Détection des voies
    const int max_lanes = min(4, actual_lanes);
    for (int lane_num = 0; lane_num < max_lanes; ++lane_num) {
        vector<cv::Point> lane_points;
        int valid_points = 0;
        for (int p = 0; p < actual_lanes; ++p) {
            if (loc[lane_num][p] > 0) {
                valid_points++;
            }
        }
    if (valid_points > 2) {
        lanes_detected[lane_num] = true;
        for (int point_num = 0; point_num < actual_lanes; ++point_num) {
            
            if (loc[point_num][lane_num] > 0) {
                // Calcul de X - formule corrigée
                float x_f = ((loc[point_num][lane_num] * col_sample_w) * cfg.img_w/800.0f)- 1.0f;
                int x = static_cast<int>(round(x_f));
                // Calcul de Y - formule corrigée
                int row_idx = cfg.cls_num_per_lane -1 - point_num;
                float y_f = (cfg.img_h * (cfg.row_anchor[row_idx]/288.0f))- 1;  // Utilisation directe des row_anchor
                int y = static_cast<int>(round(y_f));                   
                lane_points.emplace_back(x, y);
            }
        }
    } else {
        lanes_detected[lane_num] = false;
    }
    
    lane_points_mat.push_back(lane_points);
}
return {lane_points_mat, lanes_detected};
}

cv::Mat UltrafastLaneDetector::drawLanes(const cv::Mat& input_img,
    const vector<vector<cv::Point>>& lane_points_mat,
    const vector<bool>& lanes_detected,
    bool draw_points) {
        // Redimensionnement identique à Python avec INTER_AREA
        cv::Mat visualization_img;
        cv::resize(input_img, visualization_img, cv::Size(cfg.img_w, cfg.img_h), 0, 0, cv::INTER_AREA);
        // Dessin du segment de voie (identique à Python)
        if (lanes_detected.size() > 2 && lanes_detected[1] && lanes_detected[2]) {
        // Création du polygone combiné
        vector<cv::Point> polygon_points;

        // Points de la lane 1 dans l'ordre
        polygon_points.insert(polygon_points.end(), 
                    lane_points_mat[1].begin(), 
                    lane_points_mat[1].end());

        // Points de la lane 2 dans l'ordre inverse
        polygon_points.insert(polygon_points.end(), 
                    lane_points_mat[2].rbegin(), 
                    lane_points_mat[2].rend());

        // Création d'une image temporaire pour le remplissage
        cv::Mat lane_segment_img = visualization_img.clone();

        // Conversion en format attendu par fillPoly
        vector<vector<cv::Point>> polygons = {polygon_points};

        // Remplissage avec la même couleur (123,191,0) BGR
        cv::fillPoly(lane_segment_img, polygons, cv::Scalar(123, 191, 0));

        // Fusion avec la même pondération qu'en Python (0.7 et 0.3)
        cv::addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0, visualization_img);
        }

        // Dessin des points si demandé (identique à Python)
        if (draw_points) {
            for (size_t lane_num = 0; lane_num < lane_points_mat.size(); ++lane_num) {
                    for (const auto& point : lane_points_mat[lane_num]) {
                    // Même couleur que Python (BGR)
                    cv::Scalar color = lane_colors[0];
                    // Même taille de cercle (rayon 3)
                    cv::circle(visualization_img, point, 3, color, -1);
                    }
            }
        }

    return visualization_img;
}

// Instanciation explicite des templates
template class UltrafastLaneDetector::Tensor3D<float>;
template std::vector<float> UltrafastLaneDetector::arange<float>(float, float, float);