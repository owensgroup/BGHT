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
#include <cooperative_groups.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <detail/benchmark_metrics.cuh>
#include <detail/bucket.cuh>
#include <detail/kernels.cuh>
#include <detail/rng.hpp>
#include <iterator>
#include <random>

namespace bght {
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::bcht(std::size_t capacity,
                                                        Key empty_key_sentinel,
                                                        T empty_value_sentinel,
                                                        Allocator const& allocator)
    : capacity_{std::max(capacity, std::size_t{1})}
    , sentinel_key_{empty_key_sentinel}
    , sentinel_value_{empty_value_sentinel}
    , allocator_{allocator}
    , atomic_pairs_allocator_{allocator}
    , pool_allocator_{allocator}
    , size_type_allocator_{allocator} {
  // capacity_ must be multiple of bucket size
  auto remainder = capacity_ % bucket_size;
  if (remainder) {
    capacity_ += (bucket_size - remainder);
  }
  num_buckets_ = capacity_ / bucket_size;
  d_table_ = std::allocator_traits<atomic_pair_allocator_type>::allocate(
      atomic_pairs_allocator_, capacity_);
  table_ =
      std::shared_ptr<atomic_pair_type>(d_table_, bght::cuda_deleter<atomic_pair_type>());

  d_build_success_ =
      std::allocator_traits<pool_allocator_type>::allocate(pool_allocator_, 1);
  build_success_ = std::shared_ptr<bool>(d_build_success_, bght::cuda_deleter<bool>());

  value_type empty_pair{sentinel_key_, sentinel_value_};

  thrust::fill(thrust::device, d_table_, d_table_ + capacity_, empty_pair);

  // maximum number of cuckoo chains
  double lg_input_size = (float)(log((double)capacity) / log(2.0));
  const unsigned max_iter_const = 7;
  max_cuckoo_chains_ = static_cast<uint32_t>(max_iter_const * lg_input_size);

  std::mt19937 rng(2);
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);
  hf2_ = initialize_hf<hasher>(rng);

  bool success = true;
  cuda_try(cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::bcht(const bcht& other)
    : capacity_(other.capacity_)
    , sentinel_key_(other.sentinel_key_)
    , sentinel_value_(other.sentinel_value_)
    , allocator_(other.allocator_)
    , atomic_pairs_allocator_(other.atomic_pairs_allocator_)
    , pool_allocator_(other.pool_allocator_)
    , d_table_(other.d_table_)
    , table_(other.table_)
    , d_build_success_(other.d_build_success_)
    , build_success_(other.build_success_)
    , max_cuckoo_chains_(other.max_cuckoo_chains_)
    , hf0_(other.hf0_)
    , hf1_(other.hf1_)
    , hf2_(other.hf2_)
    , num_buckets_(other.num_buckets_) {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::~bcht() {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
void bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::clear() {
  value_type empty_pair{sentinel_key_, sentinel_value_};
  thrust::fill(thrust::device, d_table_, d_table_ + capacity_, empty_pair);
  bool success = true;
  cuda_try(cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
template <typename InputIt>
bool bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert(InputIt first,
                                                               InputIt last,
                                                               cudaStream_t stream) {
  const auto num_keys = std::distance(first, last);

  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
  detail::kernels::tiled_insert_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, *this);
  bool success;
  cuda_try(cudaMemcpyAsync(
      &success, d_build_success_, sizeof(bool), cudaMemcpyDeviceToHost, stream));
  return success;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B>
template <typename InputIt, typename OutputIt>
void bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find(InputIt first,
                                                             InputIt last,
                                                             OutputIt output_begin,
                                                             cudaStream_t stream) {
  const auto num_keys = last - first;
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;

  detail::kernels::tiled_find_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, output_begin, *this);
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
template <typename tile_type>
__device__ bool bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert(
    value_type const& pair,
    tile_type const& tile) {
  detail::mars_rng_32 rng;

  auto bucket_id = hf0_(pair.first) % num_buckets_;

  uint32_t cuckoo_counter = 0;
  auto lane_id = tile.thread_rank();
  const int elected_lane = 0;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type insertion_pair = pair;
  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;
  do {
    bucket_type cur_bucket(&d_table_[bucket_id * bucket_size], tile);
    cur_bucket.load(cuda::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    int load = cur_bucket.compute_load(sentinel_pair);

    if (load != bucket_size) {
      // bucket is not full
      bool cas_success = false;
      if (lane_id == elected_lane) {
        cas_success = cur_bucket.strong_cas_at_location(insertion_pair,
                                                        load,
                                                        sentinel_pair,
                                                        cuda::memory_order_relaxed,
                                                        cuda::memory_order_relaxed);
      }
      cas_success = tile.shfl(cas_success, elected_lane);
      if (cas_success) {
        return true;
      }
    } else {
      // cuckoo
      // note that if cuckoo hashing failed we might insert the key
      // but we exchanged another key
      // note that we don't need to shuffle the key since we use the same elected lane for
      // insertion
      if (lane_id == elected_lane) {
        auto random_location = rng() % bucket_size;
        auto old_pair = cur_bucket.exch_at_location(
            insertion_pair, random_location, cuda::memory_order_relaxed);

        auto bucket0 = hf0_(old_pair.first) % num_buckets_;
        auto bucket1 = hf1_(old_pair.first) % num_buckets_;
        auto bucket2 = hf2_(old_pair.first) % num_buckets_;

        auto new_bucket_id = bucket0;
        new_bucket_id = bucket_id == bucket1 ? bucket2 : new_bucket_id;
        new_bucket_id = bucket_id == bucket0 ? bucket1 : new_bucket_id;

        bucket_id = new_bucket_id;

        insertion_pair = old_pair;
      }
      bucket_id = tile.shfl(bucket_id, elected_lane);
      cuckoo_counter++;
    }
  } while (cuckoo_counter < max_cuckoo_chains_);
  return false;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
template <typename tile_type>
__device__ bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::mapped_type
bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::find(key_type const& key,
                                                              tile_type const& tile) {
  const int num_hfs = 3;
  auto bucket_id = hf0_(key) % num_buckets_;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;
  for (int hf = 0; hf < num_hfs; hf++) {
    bucket_type cur_bucket(&d_table_[bucket_id * bucket_size], tile);
    cur_bucket.load(cuda::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    int key_location = cur_bucket.find_key_location(key, key_equal{});
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location);
      return found_value;
    } else if (cur_bucket.compute_load(sentinel_pair) < bucket_size) {
      return sentinel_value_;
    } else {
      bucket_id = hf == 0 ? hf1_(key) % num_buckets_ : hf2_(key) % num_buckets_;
    }
  }

  return sentinel_value_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
template <typename RNG>
void bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::randomize_hash_functions(
    RNG& rng) {
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);
  hf2_ = initialize_hf<hasher>(rng);
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
typename bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::size_type
bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::size(cudaStream_t stream) {
  const auto sentinel_key{sentinel_key_};

  auto d_count = std::allocator_traits<size_type_allocator_type>::allocate(
      size_type_allocator_, static_cast<size_type>(1));
  cuda_try(cudaMemsetAsync(d_count, 0, sizeof(std::size_t), stream));
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (capacity_ + block_size - 1) / block_size;

  detail::kernels::count_kernel<block_size>
      <<<num_blocks, block_size, 0, stream>>>(sentinel_key, d_count, *this);
  std::size_t num_invalid_keys;
  cuda_try(cudaMemcpyAsync(
      &num_invalid_keys, d_count, sizeof(std::size_t), cudaMemcpyDeviceToHost));

  cudaFree(d_count);
  return capacity_ - num_invalid_keys;
}
}  // namespace bght
