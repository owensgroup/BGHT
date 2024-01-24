#!/bin/bash

cuda_arch="75" #specify your GPU SM/gencode
build_dir="build"
targets=("all")

cmake -B $build_dir -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch}
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)