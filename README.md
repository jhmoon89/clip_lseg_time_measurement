# clip_lseg_time_measurement

## trt version time measurement (mean of 1000 iterations)

1. CLIP only (output: vector) : 2.265ms
2. Lseg image feature model (Vit): 17.388ms
3. Lseg image feature model (Resnet): 14.841ms

## Setup
```
rm -rf build
mkdir build
cd build
cmake ..
make
./clip_trt
```
