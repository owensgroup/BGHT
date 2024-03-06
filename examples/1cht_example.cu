/*
 *   Copyright 2021-2024 The Regents of the University of California, Davis
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <bght/benchmark_helpers.cuh>
#include <bght/cht.hpp>
#include <bght/cmd.hpp>
#include <bght/gpu_timer.hpp>
#include <bght/perf_report.hpp>
#include <limits>
#include <type_traits>

#include "examples_common.hpp"

int main(int argc, char** argv) {
  using key_type = uint32_t;
  using value_type = uint32_t;

  auto arguments = std::vector<std::string>(argv, argv + argc);
  std::size_t num_keys =
      get_arg_value<std::size_t>(arguments, "num-keys").value_or(16ull);
  double load_factor = get_arg_value<double>(arguments, "load-factor").value_or(0.9);
  int device = get_arg_value<int>(arguments, "device").value_or(0);

  std::cout << "num-keys: " << num_keys << '\n';
  std::cout << "load-factor: " << load_factor << '\n';
  set_device(device);

  std::size_t capacity = static_cast<std::size_t>(double(num_keys) / load_factor);

  auto invalid_key = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<value_type>::max();

  using pair_type = bght::pair<key_type, value_type>;

  std::vector<key_type> h_keys;
  benchmark::generate_uniform_unique_keys(h_keys, num_keys);

  thrust::device_vector<key_type> d_keys(num_keys);
  thrust::device_vector<pair_type> d_pairs(num_keys);
  thrust::device_vector<key_type> d_queries(num_keys);
  thrust::device_vector<value_type> d_results(num_keys);

  d_keys = h_keys;

  // assign values
  auto to_pair = [] __host__ __device__(key_type x) { return pair_type{x, x * 10}; };
  thrust::transform(
      thrust::device, d_keys.begin(), d_keys.end(), d_pairs.begin(), to_pair);

  // prepare queries
  d_queries = d_keys;

  bght::cht<key_type, value_type> test(capacity, invalid_key, invalid_value);

  auto input_start = d_pairs.data().get();
  auto input_last = input_start + num_keys;

  auto queries_start = d_queries.data().get();
  auto queries_last = queries_start + num_keys;
  auto output_start = d_results.data().get();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  gpu_timer insertion_timer(stream);
  insertion_timer.start_timer();
  auto insertion_success = test.insert(input_start, input_last, stream);
  insertion_timer.stop_timer();
  auto insertion_s = insertion_timer.get_elapsed_s();

  cuda_try(cudaStreamSynchronize(stream));

  gpu_timer find_timer(stream);
  find_timer.start_timer();
  test.find(queries_start, queries_last, output_start, stream);
  find_timer.stop_timer();
  auto find_s = find_timer.get_elapsed_s();

  cuda_try(cudaDeviceSynchronize());

  // Comoute stats
  if (!insertion_success) {
    std::cout << "Insertion failed\n";
    std::terminate();
  }
  std_cout_perf_report(insertion_s, find_s, num_keys, num_keys);

  // validation
  thrust::host_vector<value_type> h_results = d_results;
  thrust::host_vector<value_type> h_queries = d_queries;
  for (std::size_t i = 0; i < num_keys; i++) {
    auto key = h_queries[i];
    auto expected_pair = to_pair(key);
    auto found_result = h_results[i];
    if (expected_pair.second != found_result) {
      std::cout << "Error: expected: " << expected_pair.second;
      std::cout << ", found: " << found_result << '\n';
      return 0;
    }
  }
  std::cout << "Success\n";
}
