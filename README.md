# clip_lseg_time_measurement

## trt version time measurement (mean of 1000 iterations)

1. CLIP only (output: vector) : 2.265ms
2. Lseg image feature model (Vit): 17.388ms
3. Lseg image feature model (Resnet): 14.841ms

## make trt file from onnx file
onnx파일이 있는 곳으로 간 다음 터미널을 열고 다음을 수행한다.
```
/usr/src/tensorrt/bin/trtexec --onnx=xxxxx.onnx --saveEngine=xxxxx.trt
```
--onnx에 내가 저장해둔 onnx파일, --saveEngine에 내가 trt파일로 저장하고픈 이름을 쓰면 된다.
그러면 같은 폴더에 trt파일이 생성된다.


## Setup
```
rm -rf build
mkdir build
cd build
cmake ..
make
./clip_trt
```
