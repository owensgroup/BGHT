#!/bin/bash

hip_arch="gfx1100" #specify your GPU architecture
build_dir="build"
build_type="Release"
targets=("all")

# rm -rf $build_dir

cmake -B $build_dir -DCMAKE_HIP_ARCHITECTURES=${hip_arch} -DCMAKE_BUILD_TYPE=${build_type}
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)