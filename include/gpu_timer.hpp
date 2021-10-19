#pragma once
#include <cuda_runtime.h>

struct gpu_timer {
  gpu_timer(cudaStream_t stream = 0) : start_{}, stop_{}, stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  void start_timer() { cudaEventRecord(start_, stream_); }
  void stop_timer() { cudaEventRecord(stop_, stream_); }
  float get_elapsed_ms() {
    compute_ms();
    return elapsed_time_;
  }

  float get_elapsed_s() {
    compute_ms();
    return elapsed_time_ * 0.001f;
  }
  ~gpu_timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  };

 private:
  void compute_ms() {
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_time_, start_, stop_);
  }
  cudaEvent_t start_, stop_;
  cudaStream_t stream_;
  float elapsed_time_ = 0.0f;
};