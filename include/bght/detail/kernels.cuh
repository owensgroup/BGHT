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

#pragma once
#include <hip/hip_cooperative_groups.h>
#include <bght/detail/hip_helpers.hpp>
#include <hip/std/atomic>
#include <rocprim/block/block_reduce.hpp>

namespace bght {
namespace detail {
namespace kernels {
template <typename InputIt, typename HashMap>
__global__ void tiled_insert_kernel(InputIt first, InputIt last, HashMap map) {
  // construct the tile
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  auto count = last - first;
  if ((thread_id - tile.thread_rank()) >= count) {
    return;
  }

  bool do_op = false;
  typename HashMap::value_type insertion_pair{};

  // load the input
  if (thread_id < count) {
    insertion_pair = first[thread_id];
    do_op = true;
  }

  bool success = true;
  // Do the insertion
  // auto work_queue = tile.ballot(do_op);
  auto work_queue = __ballot(do_op);
  while (work_queue) {
    // auto cur_rank = __ffs(work_queue) - 1;
    auto cur_rank = __ffsll(work_queue) - 1;
    auto cur_pair = detail::shuffle(insertion_pair, cur_rank, tile);
    bool insertion_success = map.insert(cur_pair, tile);

    if (tile.thread_rank() == cur_rank) {
      do_op = false;
      success = insertion_success;
    }
    // work_queue = tile.ballot(do_op);
    work_queue = __ballot(do_op);
  }

  if (!detail::all(success, success)) {
    *map.d_build_success_ = false;
  }
}

template <typename InputIt, typename HashMap>
__global__ void iht_tiled_insert_kernel(InputIt first, InputIt last, HashMap map) {
  // construct the tile
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  auto count = last - first;
  if ((thread_id - tile.thread_rank()) >= count) {
    return;
  }

  bool do_op = false;
  typename HashMap::value_type insertion_pair{};

  // load the input
  if (thread_id < count) {
    insertion_pair = first[thread_id];
    do_op = true;
  }

  bool success = true;
  // Do the insertion
  auto work_queue = detail::ballot(do_op, tile);
  while (work_queue) {
    auto cur_rank = detail::ffs(work_queue) - 1;
    auto cur_pair = detail::shuffle(insertion_pair, cur_rank, tile);
    bool insertion_success = map.insert(cur_pair, tile).second;

    if (tile.thread_rank() == cur_rank) {
      do_op = false;
      success = insertion_success;
    }
    work_queue = detail::ballot(do_op, tile);
  }

  if (!detail::all(success, tile)) {
    *map.d_build_success_ = false;
  }
}

template <typename InputIt, typename OutputIt, typename HashMap>
__global__ void tiled_find_kernel(InputIt first,
                                  InputIt last,
                                  OutputIt output_begin,
                                  HashMap map) {
  // construct the tile
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  auto count = last - first;
  if ((thread_id - tile.thread_rank()) >= count) {
    return;
  }

  bool do_op = false;
  typename HashMap::key_type find_key;
  typename HashMap::mapped_type result;

  // load the input
  if (thread_id < count) {
    find_key = first[thread_id];
    do_op = true;
  }

  // Do the insertion
  // auto work_queue = tile.ballot(do_op);
  auto work_queue = __ballot(do_op);
  while (work_queue) {
    // auto cur_rank = __ffs(work_queue) - 1;
    auto cur_rank = __ffsll(work_queue) - 1;
    auto cur_key = detail::shuffle(find_key, cur_rank, tile);

    typename HashMap::mapped_type find_result = map.find(cur_key, tile);

    if (tile.thread_rank() == cur_rank) {
      result = find_result;
      do_op = false;
    }
    // work_queue = tile.ballot(do_op);
    work_queue = __ballot(do_op);
  }

  if (thread_id < count) {
    output_begin[thread_id] = result;
  }
}

template <typename InputIt, typename HashMap>
__global__ void insert_kernel(InputIt first, InputIt last, HashMap map) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto count = last - first;

  if (thread_id < count) {
    auto insertion_pair = first[thread_id];
    bool success = map.insert(insertion_pair);
    if (!success) {
      map.set_build_success(false);
    }
  }
}

template <typename InputIt, typename OutputIt, typename HashMap>
__global__ void find_kernel(InputIt first,
                            InputIt last,
                            OutputIt output_begin,
                            HashMap map) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto count = last - first;

  if (thread_id < count) {
    auto find_key = first[thread_id];
    auto result = map.find(find_key);
    output_begin[thread_id] = result;
  }
}

template <int BlockSize, typename InputT, typename HashMap>
__global__ void count_kernel(const InputT count_key, std::size_t* count, HashMap map) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  using BlockReduce = rocprim::block_reduce<std::size_t, BlockSize>;

  __shared__ typename BlockReduce::storage_type temp_storage;

  std::size_t match = 0;
  if (thread_id < map.capacity_) {
    const auto key = map.d_table_[thread_id].load(hip::memory_order_relaxed).first;
    match = (key == count_key);
  }
  std::size_t block_num_matches{0};
  //  = BlockReduce(temp_storage).Sum(match);
  BlockReduce().reduce(match, block_num_matches, temp_storage);

  if (threadIdx.x == 0) {
    auto sum = atomicAdd((unsigned long long int*)count,
                         (unsigned long long int)block_num_matches);
  }
}

}  // namespace kernels
}  // namespace detail
}  // namespace bght
