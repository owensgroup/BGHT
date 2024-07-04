#pragma once

#include <hip/hip_runtime_api.h>

#include <iostream>

#define hip_try(call)                                                                 \
  do {                                                                                \
    hipError_t err = call;                                                            \
    if (err != hipSuccess) {                                                          \
      printf("HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

void set_device(int device_id) {
  int device_count;
  hip_try(hipGetDeviceCount(&device_count));
  hipDeviceProp_t devProp;
  if (device_id < device_count) {
    hip_try(hipSetDevice(device_id));
    hip_try(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "Device[" << device_id << "]: " << devProp.name << std::endl;
  } else {
    std::cout << "No capable HIP device found." << std::endl;
    std::terminate();
  }
}