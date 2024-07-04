cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(bght LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")

# options
option(build_benchmarks "Build benchmarks" ON)
option(build_tests "Build tests" ON)
option(build_examples "Build examples" ON)

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


# bght flags
add_library(bght INTERFACE)
add_library(bght::bght ALIAS bght)


# HIP and C++ compiler flags
set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++17,
        --std=c++17
        >)
set(HIP_FLAGS
    # $<IF:$<CXX_COMPILER_ID:MSVC>,
    # -Xcompiler=/std:c++17,
    # -Xcompiler=-std=c++17 -Wextra>
    # $<IF:$<CXX_COMPILER_ID:MSVC>,
    # -Xcompiler=/W4,
    # -Xcompiler=-Wall>
    # $<IF:$<CONFIG:Debug>,
    #     -G -src-in-ptx,
    #     -lineinfo>
	# --expt-extended-lambda
    # -extended-lambda
    # -Wno-deprecated-gpu-targets #silence deprecated architectures
    # --forward-unknown-to-host-linker
    # --forward-unknown-to-host-compiler
    # --expt-relaxed-constexpr
    # $<$<CONFIG:Release>:-Xptxas -warn-spills -res-usage>
    )


set(external_includes "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(bght_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(developer_flags INTERFACE)
target_include_directories( developer_flags
                            INTERFACE ${external_includes})
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:HIP>:${HIP_FLAGS}>)

target_link_libraries(bght
    INTERFACE $<BUILD_INTERFACE:developer_flags>
    INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>,,-lstdc++fs>)


include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage(
  NAME libhipcxx
  GITHUB_REPOSITORY maawad/libhipcxx
  GIT_TAG main
)


target_link_libraries(bght
    INTERFACE libhipcxx::libhipcxx)

# Visual studio setup
if(build_tests)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT test_types)
    # set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT test_bcht)
    # set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT exp2_BCHT)
    # set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT probes_per_technique)

    file(TO_NATIVE_PATH ${external_includes} external_includes_native)
    file(TO_NATIVE_PATH ${bght_includes} bght_includes_native)

    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
        $(VC_IncludePath)
        $(WindowsSDK_IncludePath)
        ${external_includes_native}
        ${bght_includes_native}
        )
endif()



add_subdirectory(include)
if(build_tests)
   add_subdirectory(test)
endif(build_tests)

if(build_benchmarks)
   add_subdirectory(benchmarks)
endif(build_benchmarks)



if(build_examples)
   add_subdirectory(examples)
endif(build_examples)
