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
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bght/pair.cuh>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>

namespace benchmark {

// min_key and max_key are exclusive
template <typename key_type>
void generate_uniform_unique_keys(
    std::vector<key_type>& keys,
    std::size_t num_keys,
    key_type min_key = std::numeric_limits<key_type>::min() + 1,
    key_type max_key = std::numeric_limits<key_type>::max() - 1,
    unsigned seed = 1,
    bool cache = false) {
  keys.resize(num_keys);
  std::string dataset_dir = "dataset";
  std::string dataset_name = std::to_string(num_keys) + "_" + std::to_string(seed);
  std::string dataset_path = dataset_dir + "/" + dataset_name;
  if (cache) {
    if (std::filesystem::exists(dataset_dir)) {
      if (std::filesystem::exists(dataset_path)) {
        std::cout << "Reading cached keys.." << std::endl;
        std::ifstream dataset(dataset_path, std::ios::binary);
        dataset.read((char*)keys.data(), sizeof(key_type) * num_keys);
        dataset.close();
        return;
      }
    } else {
      std::filesystem::create_directory(dataset_dir);
    }
  }
  std::random_device rd;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<key_type> uni(min_key, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) {
    unique_keys.insert(uni(rng));
  }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());

  if (cache) {
    std::cout << "Caching.." << std::endl;
    std::ofstream dataset(dataset_path, std::ios::binary);
    dataset.write((char*)keys.data(), sizeof(key_type) * num_keys);
    dataset.close();
  }
}
//
// template <typename key_type, typename value_type, typename function>
// uint64_t validate(const std::vector<key_type>& h_keys,
//                  const std::vector<key_type>& h_find_keys,
//                  const thrust::device_vector<value_type>& d_results,
//                  const uint32_t& num_keys,
//                  const value_type& sentinel_value,
//                  function to_value,
//                  float exist_ratio = 1.0f) {
//  uint64_t num_errors = 0;
//  uint64_t max_errors = 10;
//  using pair_type = bght::pair_type<key_type, value_type>;
//  auto h_results = thrust::host_vector<value_type>(d_results);
//  std::unordered_set<key_type> cpu_ref_set;
//  if (exist_ratio != 1.0f) {
//    cpu_ref_set.insert(h_keys.begin(), h_keys.begin() + num_keys);
//  }
//  for (size_t i = 0; i < num_keys; i++) {
//    key_type query_key = h_find_keys[i];
//    value_type query_result = h_results[i];
//    value_type expected_result = to_value(query_key);
//    if (exist_ratio != 1.0f) {
//      auto expected_result_ptr = cpu_ref_set.find(query_key);
//      if (expected_result_ptr == cpu_ref_set.end()) {
//        expected_result = sentinel_value;
//      }
//    }
//
//    if (query_result != expected_result) {
//      std::string message = std::string("query_key = ") + std::to_string(query_key) +
//                            std::string(", expected: ") +
//                            std::to_string(expected_result) + std::string(", found: ") +
//                            std::to_string(query_result);
//      std::cout << message << std::endl;
//      num_errors++;
//      if (num_errors == max_errors)
//        break;
//    }
//  }
//  return num_errors;
//}

template <typename key_type>
void prep_experiment_find_with_exist_ratio(float exist_ratio,
                                           std::size_t num_keys,
                                           const std::vector<key_type>& keys,
                                           std::vector<key_type>& find_keys,
                                           key_type* d_find_keys) {
  // Choose the keys over which we will search based on the
  // exist_ratio. Recall that keys.size() == 2 * num_keys.
  assert(num_keys * 2 == keys.size());
  unsigned int end_index = num_keys * (-exist_ratio + 2);
  unsigned int start_index = end_index - num_keys;

  static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;

  // Need to copy our range [start_index, end_index) from keys
  // into find_keys.
  std::fill(find_keys.begin(), find_keys.end(), EMPTY_VALUE);
  std::copy(keys.begin() + start_index, keys.begin() + end_index, find_keys.begin());
  cuda_try(cudaMemcpy(d_find_keys,
                      find_keys.data(),
                      sizeof(key_type) * find_keys.size(),
                      cudaMemcpyHostToDevice));
}

template <typename key_type>
void prep_experiment_find_with_exist_ratio(float exist_ratio,
                                           std::size_t num_keys,
                                           const thrust::device_vector<key_type>& keys,
                                           thrust::device_vector<key_type>& find_keys) {
  // Choose the keys over which we will search based on the
  // exist_ratio. Recall that keys.size() == 2 * num_keys.
  assert(num_keys * 2 == keys.size());
  unsigned int end_index = num_keys * (-exist_ratio + 2);
  unsigned int start_index = end_index - num_keys;

  static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;

  // Need to copy our range [start_index, end_index) from keys
  // into find_keys.
  thrust::fill(thrust::device, find_keys.begin(), find_keys.end(), EMPTY_VALUE);
  thrust::copy(thrust::device,
               keys.begin() + start_index,
               keys.begin() + end_index,
               find_keys.begin());
}

}  // namespace benchmark
