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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <benchmark_helpers.cuh>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <iht.hpp>
#include <limits>
#include <perf_report.hpp>
#include <rkg.hpp>
#include <type_traits>

int main(int argc, char** argv) {
  using key_type = uint32_t;
  using value_type = uint32_t;

  auto arguments = std::vector<std::string>(argv, argv + argc);
  std::size_t num_keys =
      get_arg_value<std::size_t>(arguments, "num-keys").value_or(16ull);
  double load_factor = get_arg_value<double>(arguments, "load-factor").value_or(0.9);
  int device = get_arg_value<int>(arguments, "device").value_or(0);
  float threshold = get_arg_value<float>(arguments, "threshold").value_or(0.8);
  float exist_ratio = get_arg_value<float>(arguments, "ratio").value_or(0.5);
  bool validate = get_arg_value<bool>(arguments, "validate").value_or(true);

  std::cout << "num-keys: " << num_keys << '\n';
  std::cout << "load-factor: " << load_factor << '\n';
  std::cout << "threshold: " << threshold << '\n';
  std::cout << "exist_ratio: " << exist_ratio << '\n';
  bght::set_device(device);

  std::size_t capacity = double(num_keys) / load_factor;

  auto invalid_key = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<value_type>::max();

  using pair_type = bght::pair<key_type, value_type>;

  std::vector<key_type> keys;
  benchmark::generate_uniform_unique_keys(keys, 2 * num_keys);

  thrust::device_vector<key_type> d_keys(keys);
  thrust::device_vector<pair_type> d_pairs(num_keys * 2);

  // assign values
  auto to_value = [] __host__ __device__(key_type x) { return pair_type{x, x * 10}; };
  thrust::transform(
      thrust::device, d_keys.begin(), d_keys.end(), d_pairs.begin(), to_value);

  // Prepare gpu memory for query testing
  //
  std::vector<key_type> find_keys(num_keys, invalid_key);
  std::vector<value_type> find_results(num_keys, invalid_value);

  thrust::device_vector<key_type> d_find_keys(num_keys, invalid_key);
  thrust::device_vector<value_type> d_find_results(num_keys, invalid_value);

  // reference set
  //
  std::unordered_set<key_type> cpu_ref_set;
  if (validate) {
    cpu_ref_set.insert(keys.begin(), keys.begin() + num_keys);
  }

  iht16<key_type, value_type, 12> test_80(capacity, invalid_key, invalid_value);  // t=80%
  iht16<key_type, value_type, 14> test_90(capacity, invalid_key, invalid_value);  // t=90%

  // Insert experiment
  //
  gpu_timer insertion_timer;
  insertion_timer.start_timer();
  bool insertion_success =
      test_80.insert(d_pairs.data().get(), d_pairs.data().get() + num_keys);
  insertion_timer.stop_timer();

  // Comoute stats
  if (!insertion_success) {
    std::cout << "Insertion failed\n";
    std::terminate();
  }

  benchmark::prep_experiment_find_with_exist_ratio(
      exist_ratio, num_keys, d_keys, d_find_keys);

  // run the find
  gpu_timer find_timer;
  find_timer.start_timer();
  test_80.find(d_find_keys.data().get(),
               d_find_keys.data().get() + num_keys,
               d_find_results.data().get());
  find_timer.stop_timer();

  auto insertion_s = insertion_timer.get_elapsed_s();
  auto find_s = find_timer.get_elapsed_s();

  if (validate) {
    // get the results back on host
    cuda_try(cudaMemcpy(find_results.data(),
                        d_find_results.data().get(),
                        sizeof(value_type) * find_keys.size(),
                        cudaMemcpyDeviceToHost));

    cuda_try(cudaMemcpy(find_keys.data(),
                        d_find_keys.data().get(),
                        sizeof(key_type) * find_keys.size(),
                        cudaMemcpyDeviceToHost));

    // Error checking and validation
    //
    std::size_t num_errors = 0;
    std::size_t found_count = 0;

    for (std::size_t key_id = 0; key_id < find_keys.size(); key_id++) {
      auto find_key = find_keys[key_id];
      auto found_result = find_results[key_id];
      auto expected_result_ptr = cpu_ref_set.find(find_key);
      value_type expected_value = invalid_value;
      if (expected_result_ptr != cpu_ref_set.end()) {
        expected_value = to_value(find_key).second;
      }
      if (expected_value != invalid_value) {
        found_count++;
      }
      if (found_result != expected_value) {
        std::cout << "Error: key " << find_key;
        std::cout << ", expected value:" << expected_value;
        std::cout << " found value:" << found_result;
        std::cout << std::endl;
        num_errors++;
      }
    }

    std::cout << "==================================\n";
    std::cout << "Find ratio was : " << float(found_count) / float(num_keys)
              << ", for exist_ratio = " << exist_ratio << "\n";

    if (num_errors != 0) {
      std::cout << " validation failed" << std::endl;
    }
  }

  std_cout_perf_report(insertion_s, find_s, num_keys, num_keys);
}
