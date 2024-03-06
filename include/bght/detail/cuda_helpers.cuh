/*
 *   Copyright 2021 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once
#include <iostream>

namespace bght {
#define _device_ __device__ __forceinline__
#define _host_device_ __device__ __host__ __forceinline__
#define _kernel_ __global__
#define DEVICE_QUALIFIER __device__ inline
namespace detail {
#define cuda_try(call)                                                                  \
  do {                                                                                  \
    cudaError_t err = call;                                                             \
    if (err != cudaSuccess) {                                                           \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::terminate();                                                                 \
    }                                                                                   \
  } while (0)

_device_ void cuda_assert(bool expression_result, char* message = nullptr) {
  if (!expression_result) {
    if (message && (threadIdx.x & 0x1f == 0)) {
      printf("assert failed: %s", message);
    }
    //__trap();
    asm("trap;");
  }
}
}  // namespace detail
}  // namespace bght
