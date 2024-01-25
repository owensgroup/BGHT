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
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
namespace rkg {
template <typename key_type, typename value_type>
value_type generate_value(key_type in) {
  return in + 1;
}

template <typename key_type, typename value_type, typename size_type>
void generate_uniform_unique_pairs(std::vector<key_type>& keys,
                                   std::vector<value_type>& values,
                                   size_type num_keys,
                                   bool cache = false,
                                   key_type min_key = 0) {
  keys.resize(num_keys);
  values.resize(num_keys);
  unsigned seed = 1;
  // bool cache = true;
  std::string dataset_dir = "dataset";
  std::string dataset_name = std::to_string(num_keys) + "_" + std::to_string(seed);
  std::string dataset_path = dataset_dir + "/" + dataset_name;
  if (cache) {
    if (std::filesystem::exists(dataset_dir)) {
      if (std::filesystem::exists(dataset_path)) {
        std::cout << "Reading cached keys.." << std::endl;
        std::ifstream dataset(dataset_path, std::ios::binary);
        dataset.read((char*)keys.data(), sizeof(key_type) * num_keys);
        dataset.read((char*)values.data(), sizeof(value_type) * num_keys);
        dataset.close();
        return;
      }
    } else {
      std::filesystem::create_directory(dataset_dir);
    }
  }
  std::random_device rd;
  std::mt19937 rng(seed);
  auto max_key = std::numeric_limits<key_type>::max() - 1;
  std::uniform_int_distribution<key_type> uni(min_key, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) {
    unique_keys.insert(uni(rng));
    // unique_keys.insert(unique_keys.size() + 1);
  }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());
  std::shuffle(keys.begin(), keys.end(), rng);

#ifdef _WIN32
  // OpenMP + windows don't allow unsigned loops
  for (uint32_t i = 0; i < unique_keys.size(); i++) {
    values[i] = generate_value<key_type, value_type>(keys[i]);
  }
#else

  for (uint32_t i = 0; i < unique_keys.size(); i++) {
    values[i] = generate_value<key_type, value_type>(keys[i]);
  }
#endif

  if (cache) {
    std::cout << "Caching.." << std::endl;
    std::ofstream dataset(dataset_path, std::ios::binary);
    dataset.write((char*)keys.data(), sizeof(key_type) * num_keys);
    dataset.write((char*)values.data(), sizeof(value_type) * num_keys);
    dataset.close();
  }
}

template <typename key_type, typename size_type>
void generate_uniform_unique_keys(std::vector<key_type>& keys, size_type num_keys) {
  keys.resize(num_keys);
  unsigned seed = 1;
  std::random_device rd;
  std::mt19937 rng(seed);
  auto max_key = std::numeric_limits<key_type>::max() - 1;
  std::uniform_int_distribution<key_type> uni(0, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) {
    unique_keys.insert(uni(rng));
  }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());
  std::shuffle(keys.begin(), keys.end(), rng);
}
}  // namespace rkg
