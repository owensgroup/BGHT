#!/bin/bash

hip_arch="gfx1100" #specify your GPU SM/gencode
build_dir="build"
targets=("all")

# rm -rf $build_dir

cmake -B $build_dir -DCMAKE_HIP_ARCHITECTURES=${hip_arch}
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)