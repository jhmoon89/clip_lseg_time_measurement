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

using namespace nvinfer1;
using namespace std::chrono;

// Custom Logger class for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Display info and warning messages during development, ignore them otherwise
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "TensorRT error: " << msg << std::endl;
        }
    }
};

// Instantiate a global logger
Logger gLogger;

int main() {
    // Initialize TensorRT runtime with the global logger
    IRuntime* runtime = createInferRuntime(gLogger);

    // Load the serialized TensorRT model
    // CLIP
    // std::ifstream file("../visual_component.trt", std::ios::binary | std::ios::ate);

    // Lseg
    // std::ifstream file("../lseg_visual_cuda12.trt", std::ios::binary | std::ios::ate);
    // std::ifstream file("../lseg_model.trt", std::ios::binary | std::ios::ate);
    // std::ifstream file("../lseg_model_partial.trt", std::ios::binary | std::ios::ate);
    std::ifstream file("../lseg_model_241115.trt", std::ios::binary | std::ios::ate);
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        std::cout << "Model loaded successfully.\n";
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
    IExecutionContext* context = engine->createExecutionContext();

    // Input and output sizes for CLIP
    // const int inputSize = 1 * 3 * 224 * 224;
    // const int outputSize = 1024;

    // Lseg
    const int inputSize = 1 * 3 * 384 * 384;
    const int outputSize = 1 * 512 * 192 * 192;
    // const int outputSize = 1 * 512 * 36864;
    
    // Allocate device memory for input and output
    void* d_input;
    void* d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));

    // Set up input dimensions
    float* input = new float[inputSize];
    std::fill(input, input + inputSize, 1.0f); // Sample input data

    // Transfer input to device
    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up run
    context->enqueueV2(&d_input, 0, nullptr); // Use enqueueV2 for better performance

    // Timing loop
    int numRuns = 1000;
    double totalDuration = 0.0;
    for (int i = 0; i < numRuns; i++) {
        auto start = high_resolution_clock::now();
        context->enqueueV2(&d_input, 0, nullptr); // Use enqueueV2
        auto end = high_resolution_clock::now();
        totalDuration += duration_cast<microseconds>(end - start).count();
    }

    // Calculate and output the average time per inference in milliseconds
    std::cout << "Average time per inference: " << (totalDuration / numRuns) / 1000.0 << " milliseconds\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete[] input;

    return 0;
}