#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "ModelType.h"
#include "ModelConfig.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

class UltrafastLaneDetector {
public:
    UltrafastLaneDetector(const std::string& model_path, ModelType model_type);
    ~UltrafastLaneDetector();
    cv::Mat detectLanes(const cv::Mat& image, bool draw_points = true);

private:
    template<typename T>
    class Tensor3D {
    public:
        Tensor3D(size_t dim1, size_t dim2, size_t dim3);
        T& operator()(size_t i, size_t j, size_t k);
        void reshape(size_t new_dim1, size_t new_dim2, size_t new_dim3);
        std::vector<T>& getData() { return data; }
        const std::vector<T>& getData() const { return data; }
        
    private:
        std::array<size_t, 3> dims;
        std::vector<T> data;
    };

    void initializeModel(const std::string& model_path);
    std::vector<std::vector<std::vector<float>>> reverse_second_dim(
        const float* output,
        int griding_num,
        int num_lanes,
        int points_per_lane);
    std::vector<std::vector<std::vector<float>>> softmax_axis0(
        const std::vector<std::vector<std::vector<float>>>& input);
    std::vector<std::vector<int>> argmax_axis0_3d(const std::vector<std::vector<std::vector<float>>>& processed_output);
    std::vector<float> prepareInput(const cv::Mat& image);
    std::pair<std::vector<std::vector<cv::Point>>, std::vector<bool>> processOutput(const float* output);
    cv::Mat drawLanes(const cv::Mat& input_img, 
                     const std::vector<std::vector<cv::Point>>& lane_points,
                     const std::vector<bool>& lanes_detected,
                     bool draw_points);

    template<typename T>
    std::vector<T> arange(T start, T stop, T step = 1);

    std::vector<std::vector<std::vector<float>>> reshapeTensor3D(
        const std::vector<float>& data,
        size_t dim1, size_t dim2, size_t dim3);

    ModelConfig cfg;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int input_height;
    int input_width;
    int channels;
    std::vector<cv::Scalar> lane_colors = {
        cv::Scalar(0,0,255),
        cv::Scalar(0,255,0),
        cv::Scalar(255,0,0),
        cv::Scalar(0,255,255)
    };
};