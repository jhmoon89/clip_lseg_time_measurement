# clip_lseg_time_measurement

## trt version time measurement (mean of 1000 iterations)

1. CLIP only (output: vector) : 2.265ms
2. Lseg image feature model (Vit): 17.388ms
3. Lseg image feature model (Resnet): 14.841ms

## make trt file from onnx file
Navigate to the directory where your ONNX file is located, open a terminal, and execute the following command:
```
/usr/src/tensorrt/bin/trtexec --onnx=xxxxx.onnx --saveEngine=xxxxx.trt
```
Replace xxxxx.onnx with the name of your ONNX file and xxxxx.trt with the desired name for your TensorRT file. The resulting .trt file will be created in the same folder.


## Setup
```
rm -rf build
mkdir build
cd build
cmake ..
make
./clip_trt
```
