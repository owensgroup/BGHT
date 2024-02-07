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
#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bght/detail/benchmark_metrics.cuh>
#include <bght/detail/bucket.cuh>
#include <bght/detail/kernels.cuh>
#include <bght/detail/rng.hpp>
#include <bght/iht.hpp>
#include <iterator>
#include <random>

namespace bght {
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iht(
    std::size_t capacity,
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
  capacity_ = detail::get_valid_capacity<bucket_size>(capacity_);
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

  hf0_ = hasher(0);
  hf1_ = hasher(1);
  hfp_ = hasher(2);

  bool success = true;
  cuda_try(cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B,
          int Threshold>
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iht(const iht& other)
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
    , hfp_(other.hfp_)
    , hf0_(other.hf0_)
    , hf1_(other.hf1_)
    , num_buckets_(other.num_buckets_) {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::~iht() {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
void iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::clear() {
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
          int B,
          int Threshold>
template <typename InputIt>
bool iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::insert(
    InputIt first,
    InputIt last,
    cudaStream_t stream) {
  const auto num_keys = std::distance(first, last);
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
  detail::kernels::iht_tiled_insert_kernel<<<num_blocks, block_size, 0, stream>>>(
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
          int B,
          int Threshold>
template <typename InputIt, typename OutputIt>
void iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::find(
    InputIt first,
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
          int B,
          int Threshold>
template <typename tile_type>
__device__ cuda::std::pair<
    typename bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iterator,
    bool>
bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::insert(
    value_type const& pair,
    tile_type const& tile) {
  auto primary_bucket = hfp_(pair.first) % num_buckets_;
  auto lane_id = tile.thread_rank();
  const int elected_lane = 0;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};

  if (key_equal{}(pair.first, sentinel_key_)) {
    return {end(), false};
  }

  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;

  int load = 0;
  bucket_type bucket(&d_table_[primary_bucket * bucket_size], tile);
  if (threshold_ > 0) {
    bucket.load(cuda::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    auto key_location = bucket.find_key_location(pair.first, key_equal{});
    // check if key exists
    if (key_location != -1) {
      return {bucket.begin() + key_location, false};
    }
    load = bucket.compute_load(sentinel_pair);
  }
  do {
    if (load >= threshold_) {
      // Secondary hashing scheme
      // Using Double hashing
      auto bucket_id = hf0_(pair.first) % num_buckets_;
      auto step_size = (hf1_(pair.first) % (capacity_ / bucket_size - 1) + 1);
      while (true) {
        bucket = bucket_type(&d_table_[bucket_id * bucket_size], tile);
        bucket.load(cuda::memory_order_relaxed);

        // check if key exists
        auto key_location = bucket.find_key_location(pair.first, key_equal{});
        if (key_location != -1) {
          return {bucket.begin() + key_location, false};
        }

        load = bucket.compute_load(sentinel_pair);
        INCREMENT_PROBES_IN_TILE
        while (load < bucket_size) {
          bool cas_success = false;
          bool key_exists = false;
          if (lane_id == elected_lane) {
            auto found =
                bucket.strong_cas_at_location_ret_old(pair,
                                                      load,
                                                      sentinel_pair,
                                                      cuda::memory_order_relaxed,
                                                      cuda::memory_order_relaxed);
            if (found.first == pair.first) {
              key_exists = true;
            }
            if (found.first == sentinel_pair.first) {
              cas_success = true;
            }
          }
          cas_success = tile.shfl(cas_success, elected_lane);
          key_exists = tile.shfl(cas_success, elected_lane);
          if (cas_success || key_exists) {
            return {bucket.begin() + load, cas_success};
          }
          load++;
        }
        bucket_id = (bucket_id + step_size) % num_buckets_;
      }
    } else {
      bool cas_success = false;
      bool key_exists = false;
      if (lane_id == elected_lane) {
        auto found = bucket.strong_cas_at_location_ret_old(pair,
                                                           load,
                                                           sentinel_pair,
                                                           cuda::memory_order_relaxed,
                                                           cuda::memory_order_relaxed);
        if (found.first == pair.first) {
          key_exists = true;
        }
        if (found.first == sentinel_pair.first) {
          cas_success = true;
        }
      }
      key_exists = tile.shfl(cas_success, elected_lane);
      cas_success = tile.shfl(cas_success, elected_lane);
      if (cas_success || key_exists) {
        return {bucket.begin() + load, cas_success};
      }
      load++;
    }
  } while (true);
  return {end(), false};
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B,
          int Threshold>
template <typename tile_type>
__device__ bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::mapped_type
bght::iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::find(
    key_type const& key,
    tile_type const& tile) {
  auto bucket_id = hfp_(key) % num_buckets_;
  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;

  // primary hash function
  bucket_type cur_bucket(&d_table_[bucket_id * bucket_size], tile);
  if (threshold_ > 0) {
    cur_bucket.load(cuda::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    int key_location = cur_bucket.find_key_location(key, key_equal{});
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location);
      return found_value;
    }
  }

  // double-hashing
  bucket_id = hf0_(key) % num_buckets_;
  auto initial_bucket = bucket_id;
  auto step_size = (hf1_(key) % (capacity_ / bucket_size - 1) + 1);
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  do {
    cur_bucket = bucket_type(&d_table_[bucket_id * bucket_size], tile);
    cur_bucket.load(cuda::memory_order_relaxed);
    INCREMENT_PROBES_IN_TILE
    int key_location = cur_bucket.find_key_location(key, key_equal{});
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location);
      return found_value;
    } else if (cur_bucket.compute_load(sentinel_pair) < bucket_size) {
      return sentinel_value_;
    }
    bucket_id = (bucket_id + step_size) % num_buckets_;
    if (bucket_id == initial_bucket)
      break;
  } while (true);

  return sentinel_value_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
template <typename RNG>
void iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::
    randomize_hash_functions(RNG& rng) {
  hfp_ = initialize_hf<hasher>(rng);
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::size_type
iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::size(cudaStream_t stream) {
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

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::const_iterator
    __device__ __host__
    iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::begin() const {
  return d_table_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iterator __device__
    __host__
    iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::begin() {
  return d_table_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::const_iterator
    __device__ __host__
    iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::end() const {
  return d_table_ + capacity_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::iterator __device__
    __host__
    iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::end() {
  return d_table_ + capacity_;
}
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator,
          int B,
          int Threshold>
__device__ __host__
    typename iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::size_type
    iht<Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold>::max_size() const {
  return capacity_;
}
}  // namespace bght
