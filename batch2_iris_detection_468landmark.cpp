#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <numeric>
#include <cmath>
#include <iterator>
#include "librealsense2/rs.hpp"
#include <chrono>
#include <thread>
// #include "opencv2/video/tracking.hpp"
#include "opencv2/tracking.hpp"

using namespace std::chrono;

struct Variance {
    float variance_x;
    float variance_y;
};

Variance calculateVariance(const std::vector<cv::Point2f>& points) {
    if (points.empty()) return {0.0f, 0.0f};

    // Calculate the mean
    float sum_x = 0.0f, sum_y = 0.0f;
    for (const auto& point : points) {
        sum_x += point.x;
        sum_y += point.y;
    }
    float mean_x = sum_x / points.size();
    float mean_y = sum_y / points.size();

    // Calculate the variance
    float variance_x = 0.0f, variance_y = 0.0f;
    for (const auto& point : points) {
        variance_x += (point.x - mean_x) * (point.x - mean_x);
        variance_y += (point.y - mean_y) * (point.y - mean_y);
    }
    variance_x /= points.size();
    variance_y /= points.size();

    return {variance_x, variance_y};  // Return both variances
}

// print process time
void printDurationInMilliseconds(const std::chrono::high_resolution_clock::time_point& start,
                                 const std::chrono::high_resolution_clock::time_point& stop,
                                 const std::string& label) {
    using namespace std::chrono;

    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << label << ": " << duration.count() / 1000.0 << " milliseconds" << std::endl;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}
// Helper function to ensure the rectangle stays within image bounds
cv::Rect ensure_rect_within_bounds(const cv::Rect& rect, const cv::Size& image_size) {
    int x = std::max(rect.x, 0);
    int y = std::max(rect.y, 0);
    int width = std::min(rect.width, image_size.width - x);
    int height = std::min(rect.height, image_size.height - y);
    return cv::Rect(x, y, width, height);
}

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};
Logger gLogger;
// Load TensorRT engine
std::tuple<nvinfer1::ICudaEngine*, nvinfer1::IRuntime*> loadEngine(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening engine file: " << engineFile << std::endl;
        return {nullptr, nullptr};
    }
    std::string engineData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create runtime!" << std::endl;
        return {nullptr, nullptr};
    }
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        runtime->destroy();
        return {nullptr, nullptr};
    }
    return {engine, runtime};
}

// Perform inference
void doInference(nvinfer1::IExecutionContext& context, 
                 float* input, float** outputs, int batchSize,
                 int inputElementSize, int* outputElementSizes, int numOutputBranches) {
    const int inputIndex = 0;
    const int outputIndex = 1; // First output branch index

    // Calculate sizes for batch processing
    size_t inputSize = batchSize * inputElementSize;

    // Set up the buffers
    void** buffers = new void*[numOutputBranches + 1]; // +1 for input buffer
    cudaError_t err;

    err = cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for input buffer: " << cudaGetErrorString(err) << std::endl;
        delete[] buffers; // Free memory on error
        return;
    }

    // Create a CUDA stream
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers[inputIndex]); // Free input buffer on error
        delete[] buffers; // Free memory on error
        return;
    }

    // Copy input data to device asynchronously
    err = cudaMemcpyAsync(buffers[inputIndex], input, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed for input buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers[inputIndex]); // Free input buffer on error
        cudaStreamDestroy(stream); // Destroy stream on error
        delete[] buffers; // Free memory on error
        return;
    }

    // Allocate memory for each output buffer based on their sizes
    for (int i = 0; i < numOutputBranches; ++i) {
        size_t outputSize = batchSize * outputElementSizes[i];
        err = cudaMalloc(&buffers[outputIndex + i], outputSize * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for output buffer: " << cudaGetErrorString(err) << std::endl;
            cudaFree(buffers[inputIndex]); // Free input buffer on error
            for (int j = 0; j < i; ++j) {
                cudaFree(buffers[outputIndex + j]); // Free previously allocated output buffers
            }
            cudaStreamDestroy(stream); // Destroy stream on error
            delete[] buffers; // Free memory on error
            return;
        }
    }

    // Enqueue the inference operation
    for (int i = 0; i < numOutputBranches; ++i) {
        if (!context.enqueueV2(&buffers[inputIndex], stream, nullptr)) {
            std::cerr << "enqueueV2 failed." << std::endl;
            cudaFree(buffers[inputIndex]); // Free input buffer on error
            for (int j = 0; j < numOutputBranches; ++j) {
                cudaFree(buffers[outputIndex + j]); // Free previously allocated output buffers
            }
            cudaStreamDestroy(stream); // Destroy stream on error
            delete[] buffers; // Free memory on error
            return;
        }
    }

    // Copy output data back to host asynchronously
    for (int i = 0; i < numOutputBranches; ++i) {
        size_t outputSize = batchSize * outputElementSizes[i];
        err = cudaMemcpyAsync(outputs[i], buffers[outputIndex + i], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync failed for output buffer: " << cudaGetErrorString(err) << std::endl;
            cudaFree(buffers[inputIndex]); // Free input buffer on error
            for (int j = 0; j < numOutputBranches; ++j) {
                cudaFree(buffers[outputIndex + j]); // Free previously allocated output buffers
            }
            cudaStreamDestroy(stream); // Destroy stream on error
            delete[] buffers; // Free memory on error
            return;
        }
    }

    // Synchronize the CUDA stream
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(buffers[inputIndex]); // Free input buffer on error
        for (int j = 0; j < numOutputBranches; ++j) {
            cudaFree(buffers[outputIndex + j]); // Free previously allocated output buffers
        }
        cudaStreamDestroy(stream); // Destroy stream on error
        delete[] buffers; // Free memory on error
        return;
    }

    // Free memory
    cudaFree(buffers[inputIndex]);
    for (int i = 0; i < numOutputBranches; ++i) {
        cudaFree(buffers[outputIndex + i]);
    }

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);
    
    delete[] buffers; // Free buffers array
}

int main() {
    // general parameters
    int image_width = 640;
    int image_height = 480;
    int fps = 30;

    // Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;
    // Create a configuration for configuring the pipeline with a non-default profile
    rs2::config cfg;
    // Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_COLOR, image_width, image_height, RS2_FORMAT_BGR8, fps);
    // Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);
    ////////////////////////////////////////////////////////////////
    // parameters for blazeface
    int num_anchors = 896;
    const int inputWidth = 128;
    const int inputHeight = 128;
    const int outputDim = num_anchors * 17; // (x, y, w, h, (x, y) for 6 landmarks, score) 
    int inputSize = 3 * inputWidth * inputHeight; // Adjust based on your input size
    int outputSize = outputDim; // Adjust based on your output size
    float box_margin = 10;
    float x_scale = 128.f;
    float y_scale = 128.f;
    float w_scale = 128.f;
    float h_scale = 128.f;
    const int numOutputBranches = 1;
    std::vector<int> outputElementSizes = {outputSize};
    // Prepare output buffers
    std::vector<float*> outputData(numOutputBranches); // Array to hold pointers to the data of each output vector

    // Allocate output data
    for (int i = 0; i < numOutputBranches; ++i) {
        outputData[i] = new float[outputElementSizes[i]]; // Allocate memory for each output branch
    }

    // parameters for face landmark
    // const int inputWidth_lm = 56;
    // const int inputHeight_lm = 56;
    const int inputWidth_lm = 256;
    const int inputHeight_lm = 256;
    int inputSize_lm = 3 * inputWidth_lm * inputHeight_lm;
    const int outputDim_lm = 1434; // 478 landmarks * (x, y, z)
    const int numOutputBranches_lm = 3;
    int outputSize_lm = outputDim_lm;
    // const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    // const std::vector<float> std = {0.229f, 0.224f, 0.225f};
    // std::vector<int> outputElementSizes_lm = {outputSize_lm, 1, 1};
    std::vector<int> outputElementSizes_lm = {1, outputSize_lm, 1};
    std::vector<float*> outputData_lm(numOutputBranches_lm);
    for (int i = 0; i < numOutputBranches_lm; ++i) {
        outputData_lm[i] = new float[outputElementSizes_lm[i]]; 
    }

    // parameters for iris landmark
    int batchSize = 2; // for left and right eyes
    const int inputWidth_il = 64;
    const int inputHeight_il = 64;
    int inputSize_il = 3 * inputWidth_il * inputHeight_il;
    const int outputDim_il = 228;
    std::vector<float> inputData_il_batch(batchSize * inputSize_il);
    std::vector<float> outputData_il_batch(batchSize * outputDim_il);
    ////////////////////////////////////////////////////////////////
    // 트래커를 위한 기본 설정
    cv::Rect iris_rect_right, iris_rect_left;
    std::vector<cv::Point2f> right_iris_points, left_iris_points;
    std::vector<cv::Point2f> prev_right_iris_points, prev_left_iris_points;
    /////////////////////////
    // cv::Ptr<cv::TrackerMIL> tracker_right_eye = cv::TrackerMIL::create();
    // cv::Ptr<cv::TrackerMIL> tracker_left_eye = cv::TrackerMIL::create();
    bool is_tracker_right_initialized = false;
    bool is_tracker_left_initialized = false;
    bool found_right = false;
    bool found_left = false;
    cv::Mat prev_frame;
    // int margin = 3;
    ////////////////////////////////////////////////////////////////
    // Load TensorRT engine for blazeface17
    // auto [engine, runtime] = loadEngine("blazeface17.trt");
    auto [engine, runtime] = loadEngine("../blazeface17.trt");
        if (!engine) {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create blazeface17 context!" << std::endl;
        engine->destroy();
        runtime->destroy();
        return -1;
    }
    ////////////////////////////////////////////////////////////////
    // Load engine for landmarks
    // auto [engine_lm, runtime_lm] = loadEngine("landmark.trt");
    // auto [engine_lm, runtime_lm] = loadEngine("../landmark.trt");
    auto [engine_lm, runtime_lm] = loadEngine("../facemesh_468.trt");
        if (!engine_lm) {
        std::cerr << "Failed to load engine_lm!" << std::endl;
        return -1;
    }
    nvinfer1::IExecutionContext* context_lm = engine_lm->createExecutionContext();
    if (!context_lm) {
        std::cerr << "Failed to create landmark context!" << std::endl;
        engine_lm->destroy();
        runtime_lm->destroy();
        return -1;
    }
    ////////////////////////////////////////////////////////////////
    // Load engine for irislandmarks
    // auto [engine_il, runtime_il] = loadEngine("irislandmark_batch2.trt");
    auto [engine_il, runtime_il] = loadEngine("../irislandmark_batch2.trt");
        if (!engine_il) {
        std::cerr << "Failed to load engine_lm!" << std::endl;
        return -1;
    }
    nvinfer1::IExecutionContext* context_il = engine_il->createExecutionContext();

    // batch size=2, 3 channels, 64x64 image size
    context_il->setBindingDimensions(0, nvinfer1::Dims4{2, 3, inputWidth_il, inputHeight_il});  
    if (!context_il) {
        std::cerr << "Failed to create landmark context!" << std::endl;
        engine_il->destroy();
        runtime_il->destroy();
        return -1;
    }
    ////////////////////////////////////////////////////////////////
    // Prepare anchor file
    std::ifstream file("../anchors.txt");
    if (!file) {
        std::cerr << "Unable to open file anchors.txt";
        return 1;
    }
    std::vector<std::vector<float>> anchors(num_anchors, std::vector<float>(4));
    // Read the values from the file
    for (size_t i = 0; i < num_anchors; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            file >> anchors[i][j];
        }
    }
    file.close();
    ////////////////////////////////////////////////////////////////
    // // Load file to save data
    // std::ofstream file_result;
    // // Open the file for writing.
    // file_result.open(filename);

    ////////////////////////////////////////////////////////////////
    // Continuously capture and display frames
    while (true) {
        // measure total processing time
        auto start_total = high_resolution_clock::now();

        // Non-blocking call to get frames from RealSense
        rs2::frameset frames;
        if (pipe.poll_for_frames(&frames)) {  // If frames are available, process them

            // // RealSense에서 프레임 가져오기
            // rs2::frameset frames = pipe.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();

            auto time_check = high_resolution_clock::now();
            printDurationInMilliseconds(start_total, time_check, "realsense get image");

            // cv::Mat로 변환
            const int w = color_frame.as<rs2::video_frame>().get_width();
            const int h = color_frame.as<rs2::video_frame>().get_height();
            cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat image_gray;
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
            auto time_check2 = high_resolution_clock::now();
            printDurationInMilliseconds(time_check, time_check2, "transform to cv::Mat");

            //////////////////////////////////////////////////////////////////////
            // blazeface inference
            // 전처리: 이미지 크기 조정 및 정규화
            cv::Mat image_resized;
            cv::Mat normalizedImage;
            cv::resize(image, image_resized, cv::Size(inputWidth, inputHeight)); // BlazeFace 입력 크기에 맞게 조정
            image_resized.convertTo(normalizedImage, CV_32FC3, 1.f / 127.5f, -1.f);
            auto time_check3 = high_resolution_clock::now();
            printDurationInMilliseconds(time_check2, time_check3, "normalize image");

            // 이미지 데이터를 1D float 배열로 변환
            std::vector<float> inputData(3 * inputWidth * inputHeight);
            std::vector<cv::Mat> channels(3);
            cv::split(normalizedImage, channels);  // 이미지를 3개의 채널로 분리
            for (int c = 0; c < 3; ++c) {
                std::memcpy(&inputData[c * inputHeight * inputWidth], channels[c].data, inputHeight * inputWidth * sizeof(float));
            }
            auto time_check4 = high_resolution_clock::now();
            printDurationInMilliseconds(time_check3, time_check4, "copy image to buffer");

            // Perform blazeface inference
            auto start_bf = high_resolution_clock::now();
            doInference(*context, inputData.data(), outputData.data(), 1, inputSize, outputElementSizes.data(), 1);
            
            // time stop after inference
            auto stop_bf = high_resolution_clock::now();
            printDurationInMilliseconds(start_bf, stop_bf, "Face box inference");
            
            std::vector<std::vector<float>> decoded_box(num_anchors, std::vector<float>(17));
            std::vector<cv::Rect> boxes_;
            std::vector<float> confidences_;

            for (int i=0; i < num_anchors; ++i) {
                float x_center = outputData[0][i * 17 + 0] / x_scale * anchors[i][2] + anchors[i][0];
                float y_center = outputData[0][i * 17 + 1] / y_scale * anchors[i][3] + anchors[i][1];
                float w = outputData[0][i * 17 + 2] / w_scale * anchors[i][2];
                float h = outputData[0][i * 17 + 3] / h_scale * anchors[i][3];

                decoded_box[i][1] = y_center - h / 2.;
                decoded_box[i][0] = x_center - w / 2.;
                decoded_box[i][3] = y_center + h / 2.;
                decoded_box[i][2] = x_center + w / 2.;
                boxes_.push_back(cv::Rect(decoded_box[i][0], decoded_box[i][1], w, h));
                    
                for (int k=0; k < 6; ++k) {
                    int offset = 4 + k * 2;
                    float keypoint_x = outputData[0][i * 17 + offset  ] / x_scale * anchors[i][2] + anchors[i][0];
                    float keypoint_y = outputData[0][i * 17 + offset+1] / y_scale * anchors[i][3] + anchors[i][1];
                    decoded_box[i][offset  ] = keypoint_x;
                    decoded_box[i][offset+1] = keypoint_y;
                }
                decoded_box[i][16] = sigmoid(outputData[0][i * 17 + 16]);
                confidences_.push_back(decoded_box[i][16]);
            }
            auto time_check5 = high_resolution_clock::now();
            printDurationInMilliseconds(stop_bf, time_check5, "Blaze face post processing");
            //////////////////////////////////////////////////////////////////////
            // NMS
            std::vector<int> output_indices_;
            auto start_nms = high_resolution_clock::now();
            float min_suppression_threshold = 0.3f; // 임계값을 조정 가능
            cv::dnn::NMSBoxes(boxes_, confidences_, 0.f, min_suppression_threshold, output_indices_);
            int final_index = output_indices_[0];
            std::cout << "Facebox score: " << confidences_[final_index] << std::endl;
            // std::cout << output_indices_[0] << std::endl;
            auto stop_nms = high_resolution_clock::now();
            printDurationInMilliseconds(start_nms, stop_nms, "NMS");
            //////////////////////////////////////////////////////////////////////
            //// face landmark inference ////
            // crop face
            cv::Point p1(decoded_box[final_index][0] * image_width, decoded_box[final_index][1] * image_height);
            cv::Point p2(decoded_box[final_index][2] * image_width, decoded_box[final_index][3] * image_height);
            float box_width = (decoded_box[final_index][2] - decoded_box[final_index][0]) * image_width;
            float box_height = (decoded_box[final_index][3] - decoded_box[final_index][1]) * image_height;
            float box_margin_top = 0.25 * box_height;
            float box_margin_bot = 0.25 * box_height;
            float box_margin_side = 0.25 * box_width;

            int facebox_x = std::max(static_cast<int>(decoded_box[final_index][0] * image_width - box_margin_side), 0);
            int facebox_y = std::max(static_cast<int>(decoded_box[final_index][1] * image_height - box_margin_top), 0);
            int facebox_width = 
            std::min(static_cast<int>((decoded_box[final_index][2] - decoded_box[final_index][0]) * image_width + 2 * box_margin_side), image_width - facebox_x);
            int facebox_height = 
            std::min(static_cast<int>((decoded_box[final_index][3] - decoded_box[final_index][1]) * image_height + box_margin_top + box_margin_bot), image_height - facebox_y);

            // Ensure width and height are not negative
            facebox_width = std::max(facebox_width, 0);
            facebox_height = std::max(facebox_height, 0);

            cv::Rect faceRect(facebox_x, facebox_y, facebox_width, facebox_height);
            cv::Mat face_crop = image(faceRect);

            auto time_check6 = high_resolution_clock::now();
            cv::Mat face_crop_resized;
            cv::resize(face_crop, face_crop_resized, cv::Size(inputWidth_lm, inputHeight_lm));
            std::vector<float> inputData_lm(3 * inputWidth_lm * inputHeight_lm);// 이미지 데이터를 1D float 배열로 변환
            // crop face image normalization
            cv::Mat face_crop_resized_normalized;
            // face_crop_resized.convertTo(face_crop_resized_normalized, CV_32FC3, 1.f / 127.5f, -1.f);
            face_crop_resized.convertTo(face_crop_resized_normalized, CV_32FC3, 1.f / 255.f);
            auto time_check7 = high_resolution_clock::now();
            printDurationInMilliseconds(time_check6, time_check7, "crop face image and resize");

            // Single loop to normalize and convert to tensor
            for (int h = 0; h < inputHeight_lm; ++h) {
                for (int w = 0; w < inputWidth_lm; ++w) {
                    cv::Vec3f& pixel = face_crop_resized_normalized.at<cv::Vec3f>(h, w);
                    // R channel
                    inputData_lm[h * inputWidth_lm + w] = pixel[2];
                    // G channel (shift by one image size)
                    inputData_lm[inputWidth_lm * inputHeight_lm + h * inputWidth_lm + w] 
                    = pixel[1];
                    // B channel (shift by two image sizes)
                    inputData_lm[2 * inputWidth_lm * inputHeight_lm + h * inputWidth_lm + w] 
                    = pixel[0];
                }
            }
            auto time_check8 = high_resolution_clock::now();
            printDurationInMilliseconds(time_check7, time_check8, "face landmark copy to input buffer");
            // Inference for landmark
            auto start_lm = high_resolution_clock::now();
            doInference(*context_lm, inputData_lm.data(), outputData_lm.data(), 1, inputSize_lm, outputElementSizes_lm.data(), numOutputBranches_lm);
            auto stop_lm = high_resolution_clock::now();
            printDurationInMilliseconds(start_lm, stop_lm, "Face landmark inference");


            int landmark_ind = 1;
            std::vector<cv::Point2d> face_lm;
            // for (int i=468;i< outputElementSizes_lm[landmark_ind]/3;++i) {
            for (int i=0;i< outputElementSizes_lm[landmark_ind]/3;++i) {
                // std::cout << outputData_lm[landmark_ind][i * 3 + 0] << " " <<
                //              outputData_lm[landmark_ind][i * 3 + 1] << " " <<
                //              outputData_lm[landmark_ind][i * 3 + 2] << std::endl;
                cv::Point p3(
                    outputData_lm[landmark_ind][i * 3], 
                    outputData_lm[landmark_ind][i * 3 + 1]
                );
                
                face_lm.push_back(p3);
                cv::circle(face_crop_resized, p3, 1, cv::Scalar(255, 255, 255), -1);
                // std::cout << outputData_lm[0] << " ";
                // std::cout << outputData_lm[0][0] << " ";
            }
            std::cout << "face conf: " << outputData_lm[2][0] << std::endl;
            
            // get center of initial eye contour landmarks
            // cv::Point2d right_eye_center = (
            //     face_lm[36] + face_lm[37] + face_lm[38] +
            //     face_lm[39] + face_lm[40] + face_lm[41]
            //     ) / 6.0;
            // cv::Point2d left_eye_center = (
            //     face_lm[42] + face_lm[43] + face_lm[44] + face_lm[45] + face_lm[46] + face_lm[47]
            //     ) / 6.0;
            // //////////////////////////////////////////////////////////////////////////////////
            // // iris inference
            // // Adjust and crop right eye
            // float iris_margin = 4.f;
            // float iris_w_right = face_lm[39].x - face_lm[36].x + 2 * iris_margin;
            // float iris_h_right = (
            //     - std::max(face_lm[37].y, face_lm[38].y) + 
            //     std::min(face_lm[40].y, face_lm[41].y) + 2 * iris_margin
            // );
            // float iris_window_right = std::max(iris_w_right, iris_h_right);
            // iris_w_right = iris_window_right;
            // iris_h_right = iris_window_right;

            // // Crop right eye using a rectangle based on landmarks
            // cv::Rect right_eye_rect(
            //     right_eye_center.x - iris_w_right / 2.f, 
            //     right_eye_center.y - iris_h_right / 2.f,
            //     iris_w_right, 
            //     iris_h_right
            // );

            // // Ensure the right eye rectangle is within the image bounds
            // right_eye_rect = ensure_rect_within_bounds(right_eye_rect, face_crop_resized.size());

            // // Crop and resize the right eye
            // cv::Mat right_eye = face_crop_resized(right_eye_rect);
            // cv::Mat right_eye_resized;
            // cv::resize(right_eye, right_eye_resized, cv::Size(inputWidth_il, inputHeight_il));

            // // Adjust and crop left eye
            // float iris_w_left = face_lm[45].x - face_lm[42].x + 2 * iris_margin;
            // float iris_h_left = (
            //     - std::max(face_lm[43].y, face_lm[44].y) + 
            //     std::min(face_lm[46].y, face_lm[47].y) + 2 * iris_margin
            // );
            // float iris_window_left = std::max(iris_w_left, iris_h_left);
            // iris_w_left = iris_window_left;
            // iris_h_left = iris_window_left;

            // // Crop left eye using a rectangle based on landmarks
            // cv::Rect left_eye_rect(
            //     left_eye_center.x - iris_w_left / 2.f, 
            //     left_eye_center.y - iris_h_left / 2.f,
            //     iris_w_left, 
            //     iris_h_left
            // );

            // // Ensure the left eye rectangle is within the image bounds
            // left_eye_rect = ensure_rect_within_bounds(left_eye_rect, face_crop_resized.size());

            // // Crop and resize the left eye
            // cv::Mat left_eye = face_crop_resized(left_eye_rect);
            // cv::Mat left_eye_resized;
            // cv::resize(left_eye, left_eye_resized, cv::Size(inputWidth_il, inputHeight_il));

            // auto time_check8_2 = high_resolution_clock::now();
            // // image normalization
            // cv::Mat iris_normalized_right;
            // cv::Mat iris_normalized_left;
            // // right_eye_resized.convertTo(iris_normalized_right, CV_32FC3, 1.f / 255.f);
            // // left_eye_resized.convertTo(iris_normalized_left, CV_32FC3, 1.f / 255.f);
            // right_eye_resized.convertTo(iris_normalized_right, CV_32FC3, 1.f / 127.5f, -1.f);
            // left_eye_resized.convertTo(iris_normalized_left, CV_32FC3, 1.f / 127.5f, -1.f);
            // auto time_check9 = high_resolution_clock::now();
            // printDurationInMilliseconds(time_check8_2, time_check9, "resize iris images");
            // //////////////////////////////////////////////
            // // inference
            
            // // Right eye data
            // for (int h = 0; h < inputHeight_il; ++h) {
            //     for (int w = 0; w < inputWidth_il; ++w) {
            //         cv::Vec3f& pixel = iris_normalized_right.at<cv::Vec3f>(h, w);
            //         inputData_il_batch[h * inputWidth_il + w] = pixel[2];  // B channel
            //         inputData_il_batch[inputWidth_il * inputHeight_il + h * inputWidth_il + w] = pixel[1];  // G channel
            //         inputData_il_batch[2 * inputWidth_il * inputHeight_il + h * inputWidth_il + w] = pixel[0];  // R channel
            //     }
            // }
            // // Left eye data (shifted by 1 * inputSize for batch 1)
            // for (int h = 0; h < inputHeight_il; ++h) {
            //     for (int w = 0; w < inputWidth_il; ++w) {
            //         cv::Vec3f& pixel = iris_normalized_left.at<cv::Vec3f>(h, w);
            //         inputData_il_batch[inputSize_il + h * inputWidth_il + w] = pixel[2];  // B channel
            //         inputData_il_batch[inputSize_il + inputWidth_il * inputHeight_il + h * inputWidth_il + w] = pixel[1];  // G channel
            //         inputData_il_batch[inputSize_il + 2 * inputWidth_il * inputHeight_il + h * inputWidth_il + w] = pixel[0];  // R channel
            //     }
            // }
            // auto time_check10 = high_resolution_clock::now();
            // printDurationInMilliseconds(time_check9, time_check10, "copy iris landmark input");
            // // Inference for both eyes (batch size = 2)
            // auto start_il_batch = high_resolution_clock::now();
            // doInference(*context_il, inputData_il_batch.data(), outputData_il_batch.data(), 2, inputSize_il, outputDim_il);
            // auto stop_il_batch = high_resolution_clock::now();
            // printDurationInMilliseconds(start_il_batch, stop_il_batch, "Iris landmark inference");
            // //////////////////////////////////////////////////////////////////////////////////
            // Visualize
            auto time_check11 = high_resolution_clock::now();
            // Draw face box
            cv::rectangle(image, p1, p2, cv::Scalar(255, 0, 0), 2, cv::LINE_8);

            // Draw 6 landmarks of Blazeface output
            for (int i=0;i<6;i++){
                cv::Point p(decoded_box[final_index][4 + 2 * i] * image_width, decoded_box[final_index][4 + 2 * i + 1] * image_height);
                cv::circle(image, p, 1, cv::Scalar(0, 255, 0), -1);// draw blazeface landmarks
            }

            // // Draw 68 face landmarks from face landmark network
            // for (int i=0;i<68;++i) {
            //     cv::Point p3(outputData_lm[i * 2] * inputWidth_lm, outputData_lm[i * 2 + 1] * inputHeight_lm);
            //     cv::circle(face_crop_resized, p3, 1, cv::Scalar(0, 255, 0), -1);// draw 68 landmarks
            // }
            
            // // // Draw the centers of the eyes (initial pupil position)
            // // cv::circle(face_crop_resized, left_eye_center, 2, cv::Scalar(0, 0, 255), -1); 
            // // cv::circle(face_crop_resized, right_eye_center, 2, cv::Scalar(0, 0, 255), -1); 

            // int offset = 213;
            // right_iris_points.clear();
            // left_iris_points.clear();

            // // Visualize or process results for both eyes
            // for (int i = 0; i < 5; ++i) {
            //     cv::Point p4(
            //         outputData_il_batch[offset + i * 3], 
            //         outputData_il_batch[offset + i * 3 + 1]
            //         );
            //     cv::circle(right_eye_resized, p4, 1, cv::Scalar(255, 255, 255), -1);
            //     cv::Point p5(
            //         outputData_il_batch[outputDim_il + offset + i * 3], 
            //         outputData_il_batch[outputDim_il + offset + i * 3 + 1]
            //         );
            //     cv::circle(left_eye_resized, p5, 1, cv::Scalar(255, 255, 255), -1);

            //     // compute iris on face crop image
            //     float iris_right_in_face_crop_x 
            //     = outputData_il_batch[offset + i * 3] / inputWidth_il * iris_w_right
            //     + right_eye_center.x - iris_w_right / 2.f;
            //     float iris_right_in_face_crop_y 
            //     = outputData_il_batch[offset + i * 3 + 1] / inputHeight_il * iris_h_right
            //     + right_eye_center.y - iris_h_right / 2.f;

            //     float iris_left_in_face_crop_x = (
            //         outputData_il_batch[outputDim_il + offset + i * 3] 
            //         / inputWidth_il * iris_w_left
            //         + left_eye_center.x - iris_w_left / 2.f
            //     );
                
            //     float iris_left_in_face_crop_y = (
            //         outputData_il_batch[outputDim_il + offset + i * 3 + 1] 
            //         / inputHeight_il * iris_h_left
            //         + left_eye_center.y - iris_h_left / 2.f
            //     );

            //     // plotting iris on the original image
            //     float iris_right_in_entire_img_x = 
            //     iris_right_in_face_crop_x / inputWidth_lm * facebox_width + facebox_x;
            //     float iris_right_in_entire_img_y = 
            //     iris_right_in_face_crop_y / inputHeight_lm * facebox_height + facebox_y;
            //     float iris_left_in_entire_img_x = 
            //     iris_left_in_face_crop_x / inputWidth_lm * facebox_width + facebox_x;
            //     float iris_left_in_entire_img_y = 
            //     iris_left_in_face_crop_y / inputHeight_lm * facebox_height + facebox_y;

            //     cv::Point p6(iris_right_in_entire_img_x, iris_right_in_entire_img_y);
            //     cv::Point p7(iris_left_in_entire_img_x, iris_left_in_entire_img_y);

            //     right_iris_points.push_back(cv::Point2f(p6.x, p6.y));    // push p6 for right eye
            //     left_iris_points.push_back(cv::Point2f(p7.x, p7.y));     // push p7 for left eye

            //     cv::circle(image, p6, 1, cv::Scalar(255, 255, 255), -1);
            //     cv::circle(image, p7, 1, cv::Scalar(255, 255, 255), -1);
            // }
            // /////////////////////////////////////////////////////
            // // Optical Flow based OpenCV Tracker
            // if (prev_frame.empty()) {
            //     prev_right_iris_points = right_iris_points;
            //     is_tracker_right_initialized = true;
            //     prev_left_iris_points = left_iris_points;
            //     is_tracker_left_initialized = true;
            //     prev_frame = image_gray.clone();
            // }
            // else {
                
            //     // Optical Flow 계산
            //     std::vector<uchar> status_right, status_left; // 각 포인트의 추적 성공 여부
            //     std::vector<float> err_right, err_left; // 에러 값
            //     std::vector<cv::Point2f> new_right_iris_points, new_left_iris_points;
            //     float var_th_min = 4.0f;
            //     float var_th_max = 20.f;
            //     if (is_tracker_right_initialized) {
            //         // Optical Flow 계산
            //         cv::calcOpticalFlowPyrLK(prev_frame, image_gray, prev_right_iris_points, new_right_iris_points, status_right, err_right);
            //         // points들의 bounding box를 계산해서 points들이 얼마나 뭉쳐있는지 확인
            //         Variance right_iris_variance = calculateVariance(new_right_iris_points);
            //         std::cout << "Right Iris Points Variance (X): " << right_iris_variance.variance_x << std::endl;
            //         std::cout << "Right Iris Points Variance (Y): " << right_iris_variance.variance_y << std::endl;
            //         if (
            //             (right_iris_variance.variance_x > var_th_min) 
            //             && (right_iris_variance.variance_y > var_th_min)
            //             && (right_iris_variance.variance_x < var_th_max)
            //             && (right_iris_variance.variance_y < var_th_max)
            //         ) {
            //             // 유효한 포인트만 업데이트
            //             for (size_t i = 0; i < status_right.size(); i++) {
            //                 if (status_right[i]) {
            //                     right_iris_points[i] = new_right_iris_points[i];
            //                 }
            //             }
            //         }
            //         // else {
            //         //     is_tracker_right_initialized = false;
            //         // }
            //     }
                
            //     if (is_tracker_left_initialized) {
            //         cv::calcOpticalFlowPyrLK(prev_frame, image_gray, prev_left_iris_points, new_left_iris_points, status_left, err_left);
            //         Variance left_iris_variance = calculateVariance(new_left_iris_points);
            //         std::cout << "Left Iris Points Variance (X): " << left_iris_variance.variance_x << std::endl;
            //         std::cout << "Left Iris Points Variance (Y): " << left_iris_variance.variance_y << std::endl;
            //         if ((left_iris_variance.variance_x > var_th_min) 
            //             && (left_iris_variance.variance_y > var_th_min)
            //             && (left_iris_variance.variance_x < var_th_max)
            //             && (left_iris_variance.variance_y < var_th_max)
            //         ) {
            //             for (size_t i = 0; i < status_left.size(); i++) {
            //                 if (status_left[i]) {
            //                     left_iris_points[i] = new_left_iris_points[i];
            //                 }
            //             }
            //         }
            //         // else {
            //         //     is_tracker_left_initialized = false;
            //         // }
            //     }

            //     // iris_rect_right = cv::boundingRect(right_iris_points);
            //     // iris_rect_left = cv::boundingRect(left_iris_points);

            //     // std::cout << "right iris bounding box width: " << iris_rect_right.width 
            //     // << ", height: " << iris_rect_right.height << std::endl;
            //     // std::cout << "left iris bounding box width: " << iris_rect_left.width 
            //     // << ", height: " << iris_rect_left.height << std::endl; 

            //     // 이전 프레임과 포인트 업데이트
            //     prev_frame = image_gray.clone();
            //     prev_right_iris_points = right_iris_points;
            //     prev_left_iris_points = left_iris_points;  

                         
            // }
            
            // // Tracked points를 이미지에 시각화
            // for (const auto& point : right_iris_points) {
            //     cv::circle(image_gray, point, 1, cv::Scalar(255, 255, 255), -1); // Red for right iris
            // }
            // for (const auto& point : left_iris_points) {
            //     cv::circle(image_gray, point, 1, cv::Scalar(255, 255, 255), -1); // Green for left iris
            // }
            // /////////////////////////////////////////////////////
            // // // Calculate bounding boxes based on the 5 points for each iris
            // // iris_rect_right = cv::boundingRect(right_iris_points);
            // // iris_rect_left = cv::boundingRect(left_iris_points);

            // // // Margin을 추가하여 사각형 크기 조정
            // // iris_rect_right.x -= margin;
            // // iris_rect_right.y -= margin;
            // // iris_rect_right.width += 2 * margin;  // 양쪽에 margin 추가
            // // iris_rect_right.height += 2 * margin; // 위아래에 margin 추가

            // // iris_rect_left.x -= margin;
            // // iris_rect_left.y -= margin;
            // // iris_rect_left.width += 2 * margin;   // 양쪽에 margin 추가
            // // iris_rect_left.height += 2 * margin;  // 위아래에 margin 추가

            // // // Initialize the KCF trackers with the initial iris bounding boxes
            // // if (!is_tracker_right_initialized) {
            // //     tracker_right_eye->init(image, iris_rect_right);
            // //     is_tracker_right_initialized = true;
            // // }
            // // if (!is_tracker_left_initialized) {
            // //     tracker_left_eye->init(image, iris_rect_left);
            // //     is_tracker_left_initialized = true;
            // // }

            // // if (is_tracker_right_initialized) {
            // //     found_right = tracker_right_eye->update(image, iris_rect_right);
            // // }

            // // if (is_tracker_left_initialized) {
            // //     found_left = tracker_left_eye->update(image, iris_rect_left);
            // // }

            // // if (found_right) {
            // //     std::cout << "tracking right iris" << std::endl;
            // //     cv::rectangle(image, iris_rect_right, cv::Scalar(255, 0, 0), 1); // Draw tracked right iris
            // // }
            // // else {
            // //     std::cout << "right iris not found..." << std::endl;
            // //     is_tracker_right_initialized = false;
            // // }

            // // if (found_left) {
            // //     std::cout << "tracking left iris" << std::endl;
            // //     cv::rectangle(image, iris_rect_left, cv::Scalar(255, 0, 0), 1);  // Draw tracked left iris
            // // }
            // // else {
            // //     std::cout << "left iris not found..." << std::endl;
            // //     is_tracker_left_initialized = false;
            // // }
            // /////////////////////////////////////////////////////

            // display windows
            cv::imshow("Display window", image);
            cv::imshow("Display cropped face", face_crop_resized);
            // cv::Mat face_final;
            // cv::resize(face_crop_resized, face_final, cv::Size(), 5.0, 5.0);
            // cv::imshow("Display cropped face", face_final);
            // cv::imshow("right eye", right_eye_resized);
            // cv::imshow("left eye", left_eye_resized);
            // cv::imshow("gray image", image_gray);
            auto time_check12 = high_resolution_clock::now();
            // printDurationInMilliseconds(time_check11, time_check12, "visualization");
            if (cv::waitKey(1) >= 0) break;  // ESC 키를 누르면 종료
            //////////////////////////////////////////////////////////////////////////////////
            auto stop_total = high_resolution_clock::now();
            printDurationInMilliseconds(start_total, stop_total, "Total time");
            std::cout << "--------------------------------------------" << std::endl;
        } else {
            // No frames available, skip processing
            // Sleep briefly to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  
        }
    }
    // Cleanup
    context->destroy();
    engine->destroy();
    runtime->destroy(); // Ensure you destroy the runtime
    context_lm->destroy();
    engine_lm->destroy();
    runtime_lm->destroy(); // Ensure you destroy the runtime
    context_il->destroy();
    engine_il->destroy();
    runtime_il->destroy();
    return 0; 
}
