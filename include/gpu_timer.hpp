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
#include <hip/hip_runtime.h>

struct gpu_timer {
  gpu_timer(hipStream_t stream = 0) : start_{}, stop_{}, stream_(stream) {
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
  }
  void start_timer() { hipEventRecord(start_, stream_); }
  void stop_timer() { hipEventRecord(stop_, stream_); }
  float get_elapsed_ms() {
    compute_ms();
    return elapsed_time_;
  }

  float get_elapsed_s() {
    compute_ms();
    return elapsed_time_ * 0.001f;
  }
  ~gpu_timer() {
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
  };

 private:
  void compute_ms() {
    hipEventSynchronize(stop_);
    hipEventElapsedTime(&elapsed_time_, start_, stop_);
  }
  hipEvent_t start_, stop_;
  hipStream_t stream_;
  float elapsed_time_ = 0.0f;
};