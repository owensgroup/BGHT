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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <unordered_set>
#include <utility>

#include <cht.hpp>

#include <thrust/device_vector.h>
#include <bcht.hpp>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <iht.hpp>
#include <p2bht.hpp>

#include <benchmark_helpers.cuh>

using key_type = uint32_t;
using value_type = uint32_t;
static constexpr key_type EMPTY_KEY = 0xFFFFFFFF;
static constexpr value_type EMPTY_VALUE = 0xFFFFFFFF;

struct bench_insert_find_result {
  float insert_time;
  std::vector<float> find_times;
};

bench_insert_find_result compute_experiment_averages(
    int num_experiments,
    float insert_time,
    const std::vector<float>& find_times) {
  float avg_insert_time = insert_time / float(num_experiments);
  std::vector<float> avg_find_times(find_times.size(), 0.0f);
  for (size_t i = 0; i < find_times.size(); i++) {
    avg_find_times[i] = find_times[i] / float(num_experiments);
  }

  return {avg_insert_time, avg_find_times};
}

template <typename HashMap,
          typename key_type,
          typename value_type,
          typename pair_type,
          typename Function>
bench_insert_find_result bench_insert_find(
    std::vector<key_type>& keys,
    thrust::device_vector<key_type>& d_keys,
    thrust::device_vector<pair_type>& d_pairs,
    thrust::device_vector<key_type>& d_find_keys,
    thrust::device_vector<key_type>& d_find_results,
    std::vector<key_type>& find_keys,
    std::vector<value_type>& find_results,
    std::unordered_set<key_type>& cpu_ref_set,
    bool validate,
    std::size_t num_keys,
    float load_factor,
    std::vector<float>& exist_ratios,
    Function to_value,
    int bucket_size) {
  assert(keys.size() == 2 * num_keys);

  std::cout << "Bench insert/search rates: " << num_keys << " keys, ";
  std::cout << "bucket_size: " << bucket_size << ", ";
  std::cout << "load_factor: " << load_factor << ", ";
  std::cout << "exist_ratios: ";
  for (auto r : exist_ratios) {
    std::cout << r << ", ";
  }
  std::cout << "\n";

  std::size_t capacity = static_cast<std::size_t>(num_keys / load_factor);
  HashMap map(capacity, EMPTY_KEY, EMPTY_VALUE);

  float insert_time = 0.0f;
  const int num_experiments = 10;

  int seed = 2;
  std::mt19937 rng(seed);

  std::vector<float> find_times(exist_ratios.size(), 0.0f);
  int exp = 0;
  uint32_t failed_count = 0;
  uint32_t max_failed_count = 50;
  while (exp < num_experiments) {
    map.clear();
    map.randomize_hash_functions(rng);

    // Insert experiment
    //
    gpu_timer insertion_timer;
    insertion_timer.start_timer();
    bool success = map.insert(d_pairs.data().get(), d_pairs.data().get() + num_keys);
    insertion_timer.stop_timer();

    if (!success) {
      failed_count++;
      if (failed_count == max_failed_count) {
        return {-1.0f, std::vector<float>(exist_ratios.size(), -1.0f)};
      }
      continue;
    }
    exp++;
    insert_time += insertion_timer.get_elapsed_s();

    // Find experiment sweep across set of exist_ratios
    //
    for (std::size_t ratio_idx = 0; ratio_idx < exist_ratios.size(); ratio_idx++) {
      // prep the find data
      benchmark::prep_experiment_find_with_exist_ratio(
          exist_ratios[ratio_idx], num_keys, d_keys, d_find_keys);

      // run the find
      gpu_timer find_timer;
      find_timer.start_timer();
      map.find(d_find_keys.data().get(),
               d_find_keys.data().get() + num_keys,
               d_find_results.data().get());
      find_timer.stop_timer();
      find_times[ratio_idx] += find_timer.get_elapsed_s();

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
          value_type expected_value = EMPTY_VALUE;
          if (expected_result_ptr != cpu_ref_set.end()) {
            expected_value = to_value(find_key).second;
          }
          if (expected_value != EMPTY_VALUE) {
            found_count++;
          }
          if (found_result != expected_value) {
            // std::cout << "Error: key " << find_key;
            // std::cout << ", expected value:" << expected_value;
            // std::cout << " found value:" << found_result;
            // std::cout << std::endl;
            num_errors++;
          }
        }

        std::cout << "==================================\n";
        std::cout << "Find ratio was : " << float(found_count) / float(num_keys)
                  << ", for exist_ratio = " << exist_ratios[ratio_idx] << "\n";

        if (num_errors != 0) {
          std::cout << " validation failed" << std::endl;
        }
      }
    }
  }

  return compute_experiment_averages(num_experiments, insert_time, find_times);
}

template <typename key_type, typename pair_type, typename Function>
void bench_bcht(std::vector<key_type>& keys,
                thrust::device_vector<key_type>& d_keys,
                thrust::device_vector<pair_type>& d_pairs,
                thrust::device_vector<key_type>& d_find_keys,
                thrust::device_vector<key_type>& d_find_results,
                std::vector<key_type>& find_keys,
                std::vector<value_type>& find_results,
                std::unordered_set<key_type>& cpu_ref_set,
                bool validate,
                std::size_t num_keys,
                float load_factor,
                Function to_value,
                std::string dir) {
  std::cout << "keys.size = " << keys.size() << ", 2 * num_keys = " << 2 * num_keys
            << "\n";

  std::string fname = "bcht_rates_fixed_keys.csv";
  bool output_file_exist = std::filesystem::exists(dir + fname);
  std::fstream output(dir + fname, std::ios::app);

  std::vector<float> exist_ratios = {1.0f, 0.5f, 0.0f};

  if (!output_file_exist) {
    // header
    output << "num_keys,load_factor,";
    output << "insert_1,";
    for (size_t i = 0; i < exist_ratios.size(); i++) {
      int exist_ratio = exist_ratios[i] * 100.0f;
      output << "find_1_" + std::to_string(exist_ratio) + ",";
    }
    output << "insert_8,";
    for (size_t i = 0; i < exist_ratios.size(); i++) {
      int exist_ratio = exist_ratios[i] * 100.0f;
      output << "find_8_" + std::to_string(exist_ratio) + ",";
    }
    output << "insert_16,";
    for (size_t i = 0; i < exist_ratios.size(); i++) {
      int exist_ratio = exist_ratios[i] * 100.0f;
      output << "find_16_" + std::to_string(exist_ratio) + ",";
    }
    output << "insert_32,";
    for (size_t i = 0; i < exist_ratios.size(); i++) {
      int exist_ratio = exist_ratios[i] * 100.0f;
      output << "find_32_" + std::to_string(exist_ratio) + ",";
    }
    output << std::endl;
  }

  output << num_keys << ",";
  output << load_factor << ",";

  // 1
  {
    auto bcht_1_result =
        bench_insert_find<bght::cht<key_type, value_type>>(keys,
                                                           d_keys,
                                                           d_pairs,
                                                           d_find_keys,
                                                           d_find_results,
                                                           find_keys,
                                                           find_results,
                                                           cpu_ref_set,
                                                           validate,
                                                           num_keys,
                                                           load_factor,
                                                           exist_ratios,
                                                           to_value,
                                                           1);
    output << float(num_keys) / 1.0e6 / bcht_1_result.insert_time << ",";
    for (size_t i = 0; i < bcht_1_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / bcht_1_result.find_times[i] << ",";
    }
  }

  // 8
  {
    auto bcht_8_result = bench_insert_find<bcht8<key_type, value_type>>(keys,
                                                                        d_keys,
                                                                        d_pairs,
                                                                        d_find_keys,
                                                                        d_find_results,
                                                                        find_keys,
                                                                        find_results,
                                                                        cpu_ref_set,
                                                                        validate,
                                                                        num_keys,
                                                                        load_factor,
                                                                        exist_ratios,
                                                                        to_value,
                                                                        8);
    output << float(num_keys) / 1.0e6 / bcht_8_result.insert_time << ",";
    for (size_t i = 0; i < bcht_8_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / bcht_8_result.find_times[i] << ",";
    }
  }

  // 16
  {
    auto bcht_16_result = bench_insert_find<bcht16<key_type, value_type>>(keys,
                                                                          d_keys,
                                                                          d_pairs,
                                                                          d_find_keys,
                                                                          d_find_results,
                                                                          find_keys,
                                                                          find_results,
                                                                          cpu_ref_set,
                                                                          validate,
                                                                          num_keys,
                                                                          load_factor,
                                                                          exist_ratios,
                                                                          to_value,
                                                                          16);
    output << float(num_keys) / 1.0e6 / bcht_16_result.insert_time << ",";
    for (size_t i = 0; i < bcht_16_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / bcht_16_result.find_times[i] << ",";
    }
  }

  // 32
  {
    auto bcht_32_result = bench_insert_find<bcht32<key_type, value_type>>(keys,
                                                                          d_keys,
                                                                          d_pairs,
                                                                          d_find_keys,
                                                                          d_find_results,
                                                                          find_keys,
                                                                          find_results,
                                                                          cpu_ref_set,
                                                                          validate,
                                                                          num_keys,
                                                                          load_factor,
                                                                          exist_ratios,
                                                                          to_value,
                                                                          32);
    output << float(num_keys) / 1.0e6 / bcht_32_result.insert_time << ",";
    for (size_t i = 0; i < bcht_32_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / bcht_32_result.find_times[i] << ",";
    }
  }
  output << std::endl;
}

template <typename key_type, typename pair_type, typename Function>
void bench_iht(std::vector<key_type>& keys,
               thrust::device_vector<key_type>& d_keys,
               thrust::device_vector<pair_type>& d_pairs,
               thrust::device_vector<key_type>& d_find_keys,
               thrust::device_vector<key_type>& d_find_results,
               std::vector<key_type>& find_keys,
               std::vector<value_type>& find_results,
               std::unordered_set<key_type>& cpu_ref_set,
               bool validate,
               std::size_t num_keys,
               float load_factor,
               Function to_value,
               std::string dir) {
  std::cout << "keys.size = " << keys.size() << ", 2 * num_keys = " << 2 * num_keys
            << "\n";

  std::string fname = "iht_rates_fixed_keys.csv";
  bool output_file_exist = std::filesystem::exists(dir + fname);
  std::fstream output(dir + fname, std::ios::app);

  std::vector<float> exist_ratios = {1.0f, 0.5f, 0.0f};
  std::vector<float> thresholds = {0.2, 0.4, 0.6, 0.8};

  if (!output_file_exist) {
    // header
    output << "num_keys,load_factor,";
    for (const auto& threshold : thresholds) {
      output << "insert_16_" + std::to_string(int(threshold * 100)) + ",";
      for (size_t i = 0; i < exist_ratios.size(); i++) {
        int exist_ratio = exist_ratios[i] * 100.0f;
        output << "find_16_" + std::to_string(int(threshold * 100)) + "_" +
                      std::to_string(exist_ratio) + ",";
      }
    }
    for (const auto& threshold : thresholds) {
      output << "insert_32_" + std::to_string(int(threshold * 100)) + ",";
      for (size_t i = 0; i < exist_ratios.size(); i++) {
        int exist_ratio = exist_ratios[i] * 100.0f;
        output << "find_32_" + std::to_string(int(threshold * 100)) + "_" +
                      std::to_string(exist_ratio) + ",";
      }
    }
    output << std::endl;
  }

  output << num_keys << ",";
  output << load_factor << ",";

  // 16  0.2
  {
    auto iht_16_result = bench_insert_find<iht16<key_type, value_type, 3>>(keys,
                                                                           d_keys,
                                                                           d_pairs,
                                                                           d_find_keys,
                                                                           d_find_results,
                                                                           find_keys,
                                                                           find_results,
                                                                           cpu_ref_set,
                                                                           validate,
                                                                           num_keys,
                                                                           load_factor,
                                                                           exist_ratios,
                                                                           to_value,
                                                                           16);
    output << float(num_keys) / 1.0e6 / iht_16_result.insert_time << ",";
    for (size_t i = 0; i < iht_16_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_16_result.find_times[i] << ",";
    }
  }

  // 16  0.4
  {
    auto iht_16_result = bench_insert_find<iht16<key_type, value_type, 6>>(keys,
                                                                           d_keys,
                                                                           d_pairs,
                                                                           d_find_keys,
                                                                           d_find_results,
                                                                           find_keys,
                                                                           find_results,
                                                                           cpu_ref_set,
                                                                           validate,
                                                                           num_keys,
                                                                           load_factor,
                                                                           exist_ratios,
                                                                           to_value,
                                                                           16);
    output << float(num_keys) / 1.0e6 / iht_16_result.insert_time << ",";
    for (size_t i = 0; i < iht_16_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_16_result.find_times[i] << ",";
    }
  }

  // 16  0.6
  {
    auto iht_16_result = bench_insert_find<iht16<key_type, value_type, 9>>(keys,
                                                                           d_keys,
                                                                           d_pairs,
                                                                           d_find_keys,
                                                                           d_find_results,
                                                                           find_keys,
                                                                           find_results,
                                                                           cpu_ref_set,
                                                                           validate,
                                                                           num_keys,
                                                                           load_factor,
                                                                           exist_ratios,
                                                                           to_value,
                                                                           16);
    output << float(num_keys) / 1.0e6 / iht_16_result.insert_time << ",";
    for (size_t i = 0; i < iht_16_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_16_result.find_times[i] << ",";
    }
  }

  // 16  0.8
  {
    auto iht_16_result =
        bench_insert_find<iht16<key_type, value_type, 12>>(keys,
                                                           d_keys,
                                                           d_pairs,
                                                           d_find_keys,
                                                           d_find_results,
                                                           find_keys,
                                                           find_results,
                                                           cpu_ref_set,
                                                           validate,
                                                           num_keys,
                                                           load_factor,
                                                           exist_ratios,
                                                           to_value,
                                                           16);
    output << float(num_keys) / 1.0e6 / iht_16_result.insert_time << ",";
    for (size_t i = 0; i < iht_16_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_16_result.find_times[i] << ",";
    }
  }

  // 32 0.2
  {
    auto iht_32_result = bench_insert_find<iht32<key_type, value_type, 6>>(keys,
                                                                           d_keys,
                                                                           d_pairs,
                                                                           d_find_keys,
                                                                           d_find_results,
                                                                           find_keys,
                                                                           find_results,
                                                                           cpu_ref_set,
                                                                           validate,
                                                                           num_keys,
                                                                           load_factor,
                                                                           exist_ratios,
                                                                           to_value,
                                                                           32);
    output << float(num_keys) / 1.0e6 / iht_32_result.insert_time << ",";
    for (size_t i = 0; i < iht_32_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_32_result.find_times[i] << ",";
    }
  }

  // 32 0.4
  {
    auto iht_32_result =
        bench_insert_find<iht32<key_type, value_type, 12>>(keys,
                                                           d_keys,
                                                           d_pairs,
                                                           d_find_keys,
                                                           d_find_results,
                                                           find_keys,
                                                           find_results,
                                                           cpu_ref_set,
                                                           validate,
                                                           num_keys,
                                                           load_factor,
                                                           exist_ratios,
                                                           to_value,
                                                           32);
    output << float(num_keys) / 1.0e6 / iht_32_result.insert_time << ",";
    for (size_t i = 0; i < iht_32_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_32_result.find_times[i] << ",";
    }
  }

  // 32 0.6
  {
    auto iht_32_result =
        bench_insert_find<iht32<key_type, value_type, 19>>(keys,
                                                           d_keys,
                                                           d_pairs,
                                                           d_find_keys,
                                                           d_find_results,
                                                           find_keys,
                                                           find_results,
                                                           cpu_ref_set,
                                                           validate,
                                                           num_keys,
                                                           load_factor,
                                                           exist_ratios,
                                                           to_value,
                                                           32);
    output << float(num_keys) / 1.0e6 / iht_32_result.insert_time << ",";
    for (size_t i = 0; i < iht_32_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_32_result.find_times[i] << ",";
    }
  }

  // 32 0.8
  {
    auto iht_32_result =
        bench_insert_find<iht32<key_type, value_type, 25>>(keys,
                                                           d_keys,
                                                           d_pairs,
                                                           d_find_keys,
                                                           d_find_results,
                                                           find_keys,
                                                           find_results,
                                                           cpu_ref_set,
                                                           validate,
                                                           num_keys,
                                                           load_factor,
                                                           exist_ratios,
                                                           to_value,
                                                           32);
    output << float(num_keys) / 1.0e6 / iht_32_result.insert_time << ",";
    for (size_t i = 0; i < iht_32_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / iht_32_result.find_times[i] << ",";
    }
  }

  output << std::endl;
}

template <typename key_type, typename pair_type, typename Function>
void bench_p2bht(std::vector<key_type>& keys,
                 thrust::device_vector<key_type>& d_keys,
                 thrust::device_vector<pair_type>& d_pairs,
                 thrust::device_vector<key_type>& d_find_keys,
                 thrust::device_vector<key_type>& d_find_results,
                 std::vector<key_type>& find_keys,
                 std::vector<value_type>& find_results,
                 std::unordered_set<key_type>& cpu_ref_set,
                 bool validate,
                 std::size_t num_keys,
                 float load_factor,
                 Function to_value,
                 std::string dir) {
  std::cout << "keys.size = " << keys.size() << ", 2 * num_keys = " << 2 * num_keys
            << "\n";

  std::string fname = "p2bht_rates_fixed_keys.csv";
  bool output_file_exist = std::filesystem::exists(dir + fname);
  std::fstream output(dir + fname, std::ios::app);

  std::vector<float> exist_ratios = {1.0f, 0.5f, 0.0f};

  if (!output_file_exist) {
    // header
    output << "num_keys,load_factor,";
    output << "insert_16,";
    for (size_t i = 0; i < exist_ratios.size(); i++) {
      int exist_ratio = exist_ratios[i] * 100.0f;
      output << "find_16_" + std::to_string(exist_ratio) + ",";
    }
    output << "insert_32,";
    for (size_t i = 0; i < exist_ratios.size(); i++) {
      int exist_ratio = exist_ratios[i] * 100.0f;
      output << "find_32_" + std::to_string(exist_ratio) + ",";
    }
    output << std::endl;
  }

  output << num_keys << ",";
  output << load_factor << ",";

  // 16
  {
    auto p2cht_16_result =
        bench_insert_find<p2bht16<key_type, value_type>>(keys,
                                                         d_keys,
                                                         d_pairs,
                                                         d_find_keys,
                                                         d_find_results,
                                                         find_keys,
                                                         find_results,
                                                         cpu_ref_set,
                                                         validate,
                                                         num_keys,
                                                         load_factor,
                                                         exist_ratios,
                                                         to_value,
                                                         16);
    output << float(num_keys) / 1.0e6 / p2cht_16_result.insert_time << ",";
    for (size_t i = 0; i < p2cht_16_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / p2cht_16_result.find_times[i] << ",";
    }
  }

  // 32
  {
    auto p2cht_32_result =
        bench_insert_find<p2bht32<key_type, value_type>>(keys,
                                                         d_keys,
                                                         d_pairs,
                                                         d_find_keys,
                                                         d_find_results,
                                                         find_keys,
                                                         find_results,
                                                         cpu_ref_set,
                                                         validate,
                                                         num_keys,
                                                         load_factor,
                                                         exist_ratios,
                                                         to_value,
                                                         16);
    output << float(num_keys) / 1.0e6 / p2cht_32_result.insert_time << ",";
    for (size_t i = 0; i < p2cht_32_result.find_times.size(); i++) {
      output << float(num_keys) / 1.0e6 / p2cht_32_result.find_times[i] << ",";
    }
  }
  output << std::endl;
}

int main(int argc, char** argv) {
  // set device
  int device_count;
  auto arguments = std::vector<std::string>(argv, argv + argc);
  int device_id = get_arg_value<int>(arguments, "device").value_or(0);
  bool validate = get_arg_value<bool>(arguments, "validate").value_or(true);
  std::size_t num_keys = get_arg_value<uint64_t>(arguments, "num-keys")
                             .value_or(512000);  // 16384;//8192;//4096;

  std::cout << "Device id = " << device_id << '\n';
  std::cout << "validate = " << std::boolalpha << validate << '\n';
  std::cout << "num_keys = " << num_keys << '\n' << '\n';

  cudaGetDeviceCount(&device_count);
  cudaDeviceProp devProp;
  if (device_count) {
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&devProp, device_id);
  } else {
    return 0;
  }

  std::string device_name(devProp.name);
  std::replace(device_name.begin(), device_name.end(), ' ', '-');
  std::cout << "Device[" << device_id << "]: " << device_name << std::endl;

  using key_type = uint32_t;
  using value_type = key_type;

  std::vector<key_type> keys;

  benchmark::generate_uniform_unique_keys(keys, 2 * num_keys);

  using pair_type = bght::pair<key_type, value_type>;
  thrust::device_vector<key_type> d_keys(keys);
  // d_keys.resize(num_keys);
  thrust::device_vector<pair_type> d_pairs(num_keys * 2);

  auto to_value = [] __host__ __device__(key_type x) { return pair_type{x, x * 10}; };

  thrust::transform(
      thrust::device, d_keys.begin(), d_keys.end(), d_pairs.begin(), to_value);

  // Prepare gpu memory for query testing
  //
  std::vector<key_type> find_keys(num_keys, EMPTY_VALUE);
  std::vector<value_type> find_results(num_keys, EMPTY_VALUE);

  thrust::device_vector<key_type> d_find_keys(num_keys, EMPTY_KEY);
  thrust::device_vector<value_type> d_find_results(num_keys, EMPTY_VALUE);

  // reference set
  //
  std::unordered_set<key_type> cpu_ref_set;
  if (validate) {
    cpu_ref_set.insert(keys.begin(), keys.begin() + num_keys);
  }

  std::string dir = "./results/" + device_name + "/" + "rates_fixed_keys/";
  std::filesystem::create_directories(dir);

  std::vector<float> load_factors = {0.60f,
                                     0.65f,
                                     0.70f,
                                     0.75f,
                                     0.80f,
                                     0.82f,
                                     0.84f,
                                     0.86f,
                                     0.88f,
                                     0.90f,
                                     0.91f,
                                     0.92f,
                                     0.93f,
                                     0.94f,
                                     0.95f,
                                     0.96f,
                                     0.97f,
                                     0.98f,
                                     0.99f};

  for (auto& load_factor : load_factors) {
    bench_bcht(keys,
               d_keys,
               d_pairs,
               d_find_keys,
               d_find_results,
               find_keys,
               find_results,
               cpu_ref_set,
               validate,
               num_keys,
               load_factor,
               to_value,
               dir);

    bench_p2bht(keys,
                d_keys,
                d_pairs,
                d_find_keys,
                d_find_results,
                find_keys,
                find_results,
                cpu_ref_set,
                validate,
                num_keys,
                load_factor,
                to_value,
                dir);

    bench_iht(keys,
              d_keys,
              d_pairs,
              d_find_keys,
              d_find_results,
              find_keys,
              find_results,
              cpu_ref_set,
              validate,
              num_keys,
              load_factor,
              to_value,
              dir);
  }

  return 0;
}
