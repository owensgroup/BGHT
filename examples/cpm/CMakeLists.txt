cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CPMTest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_SOURCE_DIR}/../../cmake/CPM.cmake)

CPMAddPackage(
  NAME bght
  GITHUB_REPOSITORY owensgroup/BGHT
  GIT_TAG main
  OPTIONS
     "build_tests OFF"
     "build_benchmarks OFF"
     "build_examples OFF"
)

set(CUDA_ARCHS 70)
add_executable(test_cpm test_cpm.cu)
target_link_libraries(test_cpm PRIVATE bght)
set_target_properties(test_cpm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})

#visual studio project setup
set(bght_includes "${bght_SOURCE_DIR}/include")
file(TO_NATIVE_PATH ${bght_includes} bght_includes_native)
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${external_includes_native}
    ${bght_includes_native}
    )
