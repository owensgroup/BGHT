#!/bin/bash

cuda_arch="70"
build_dir="build"
targets=("all")

cmake -B $build_dir -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch}
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)