/*
 *   Copyright 2024 The Regents of the University of California, Davis
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
#include <bght/cmd.hpp>
#include <bght/gpu_timer.hpp>
#include <bght/iht.hpp>
#include <bght/perf_report.hpp>
#include <bght/rkg.hpp>
#include <bght/tile_wide_queue.hpp>
#include <limits>
#include <type_traits>

template <typename HashMap>
__global__ void print(HashMap result_table) {
  using pair_type = typename HashMap::value_type;
  using key_type = typename HashMap::key_type;
  using value_type = typename HashMap::mapped_type;

  const auto capacity = result_table.max_size();
  const auto sentinel_key = result_table.get_sentinel_key();
  const auto sentinel_pair = result_table.get_sentinel_pair();

  auto begin = result_table.begin();

  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  const auto pair = thread_id < capacity
                        ? (begin + thread_id)->load(cuda::memory_order_relaxed)
                        : sentinel_pair;
  if (pair.first != sentinel_key) {
    printf("result_map[%u] = %u\n", pair.first, pair.second);
  };
}

template <typename HashMap, typename key_type>
__global__ void histogram(HashMap map, key_type* keys, std::size_t count) {
  using pair_type = typename HashMap::value_type;
  using value_type = typename HashMap::mapped_type;

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  using queue_type = bght::tile_wide_queue<key_type, decltype(tile)>;

  const auto sentinel_key = map.get_sentinel_key();

  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  const auto key = thread_id < count ? keys[thread_id] : sentinel_key;

  queue_type work_queue(key, sentinel_key, tile);

  while (!work_queue.empty()) {
    auto cur_key = work_queue.front();
    work_queue.pop();
    // Try inserting the key with count of 1
    auto result = map.insert({cur_key, 1}, tile);
    const bool success = result.second == true;
    // If insertion failed, one thread out of the tile need to try to atomically increment
    // the count.
    if (!success) {
      // make sure one thread in the tile do the increment
      if (tile.thread_rank() == 0) {
        bool exchange_success{false};
        auto expected = result.first->load(cuda::memory_order_relaxed);
        while (!exchange_success) {
          pair_type desired{expected.first, expected.second + 1};
          cuda::memory_order success = cuda::memory_order_relaxed;
          cuda::memory_order failure = cuda::memory_order_relaxed;
          // `expected` here will be modified with the latest value if the
          // `compare_exchange_strong` fails. On failure, the next iteration will add 1 to
          // the last found value in memory.
          // Similar logic can be used to implement min or max.
          exchange_success =
              result.first->compare_exchange_strong(expected, desired, success, failure);
        }
      }
    }
  }
}

int main(int, char**) {
  using key_type = std::uint32_t;
  using count_type = std::uint32_t;
  using pair_type = bght::pair<key_type, count_type>;

  std::vector<key_type> h_keys{512, 1, 1, 1, 1, 1, 1, 1, 1, 1,   2,  4,
                               2,   2, 4, 4, 4, 5, 5, 5, 5, 100, 512};

  std::cout << "Building a histogram for keys: ";
  for (const auto k : h_keys) {
    std::cout << k << ", ";
  }
  std::cout << std::endl;

  const auto num_keys = h_keys.size();

  auto invalid_key = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<count_type>::max();

  const float load_factor = 0.7;

  std::size_t capacity = double(num_keys) / load_factor;

  using hash_map = bght::iht<key_type, count_type>;

  hash_map map(capacity, invalid_key, invalid_value);

  thrust::device_vector<key_type> d_keys(h_keys);

  const uint32_t block_size = 128;
  uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
  histogram<<<num_blocks, block_size>>>(map, d_keys.data().get(), num_keys);

  cuda_try(cudaDeviceSynchronize());

  std::cout << "Found results:" << std::endl;

  num_blocks = (capacity + block_size - 1) / block_size;
  print<<<num_blocks, block_size>>>(map);

  cuda_try(cudaDeviceSynchronize());

  std::cout << "Exepcted reults: " << std::endl;
  std::unordered_map<key_type, count_type> histogram;
  for (int key : h_keys) {
    histogram[key]++;
  }

  for (const auto& pair : histogram) {
    printf("ground_truth[%u] = %u\n", pair.first, pair.second);
  }
}