#pragma once

#include <hip/hip_runtime_api.h>
#include <stdio.h>

#include <exception>

#define hip_try(call)                                                                 \
  do {                                                                                \
    hipError_t err = call;                                                            \
    if (err != hipSuccess) {                                                          \
      printf("HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)
