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
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bght/detail/benchmark_metrics.cuh>
#include <bght/detail/bucket.cuh>
#include <bght/detail/kernels.cuh>
#include <bght/detail/rng.hpp>
#include <iterator>
#include <random>

namespace bght {
template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator>
cht<Key, T, Hash, KeyEqual, Scope, Allocator>::cht(std::size_t capacity,
                                                   Key empty_key_sentinel,
                                                   T empty_value_sentinel,
                                                   Allocator const& allocator)
    : capacity_{std::max(capacity, std::size_t{1})}
    , sentinel_key_{empty_key_sentinel}
    , sentinel_value_{empty_value_sentinel}
    , allocator_{allocator}
    , atomic_pairs_allocator_{allocator}
    , pool_allocator_{allocator} {
  num_buckets_ = capacity_;
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
  hf3_ = initialize_hf<hasher>(rng);

  bool success = true;
  cuda_try(cudaMemcpy(d_build_success_, &success, sizeof(bool), cudaMemcpyHostToDevice));
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator>
cht<Key, T, Hash, KeyEqual, Scope, Allocator>::cht(const cht& other)
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
    , hf3_(other.hf3_)
    , num_buckets_(other.num_buckets_) {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator>
cht<Key, T, Hash, KeyEqual, Scope, Allocator>::~cht() {}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          typename Allocator>
void cht<Key, T, Hash, KeyEqual, Scope, Allocator>::clear() {
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
          typename Allocator>
template <typename InputIt>
bool cht<Key, T, Hash, KeyEqual, Scope, Allocator>::insert(InputIt first,
                                                           InputIt last,
                                                           cudaStream_t stream) {
  const auto num_keys = std::distance(first, last);

  const uint32_t block_size = 128;
  const uint32_t num_blocks =
      static_cast<uint32_t>((num_keys + block_size - 1) / block_size);
  detail::kernels::insert_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, *this);
  // cuda_try(cudaPeekAtLastError());

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
          typename Allocator>
template <typename InputIt, typename OutputIt>
void cht<Key, T, Hash, KeyEqual, Scope, Allocator>::find(InputIt first,
                                                         InputIt last,
                                                         OutputIt output_begin,
                                                         cudaStream_t stream) {
  const auto num_keys = std::distance(first, last);

  const uint32_t block_size = 128;
  const uint32_t num_blocks =
      static_cast<uint32_t>((num_keys + block_size - 1) / block_size);

  detail::kernels::find_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, output_begin, *this);
  // cuda_try(cudaPeekAtLastError());
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator>
__device__ bool bght::cht<Key, T, Hash, KeyEqual, Scope, Allocator>::insert(
    value_type const& pair) {
  auto bucket_id = hf0_(pair.first) % num_buckets_;

  uint32_t cuckoo_counter = 0;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type insertion_pair = pair;
  do {
    auto old_pair =
        d_table_[bucket_id].exchange(insertion_pair, cuda::memory_order_relaxed);
    INCREMENT_PROBES
    if (old_pair.first == sentinel_key_) {
      return true;
    } else {
      auto bucket0 = hf0_(old_pair.first) % num_buckets_;
      auto bucket1 = hf1_(old_pair.first) % num_buckets_;
      auto bucket2 = hf2_(old_pair.first) % num_buckets_;
      auto bucket3 = hf3_(old_pair.first) % num_buckets_;

      auto new_bucket_id = bucket0;
      new_bucket_id = bucket_id == bucket2 ? bucket3 : new_bucket_id;
      new_bucket_id = bucket_id == bucket1 ? bucket2 : new_bucket_id;
      new_bucket_id = bucket_id == bucket0 ? bucket1 : new_bucket_id;

      bucket_id = new_bucket_id;
      insertion_pair = old_pair;
    }
    cuckoo_counter++;
  } while (cuckoo_counter < max_cuckoo_chains_);
  return false;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator>
__device__ bght::cht<Key, T, Hash, KeyEqual, Scope, Allocator>::mapped_type
bght::cht<Key, T, Hash, KeyEqual, Scope, Allocator>::find(key_type const& key) {
  const int num_hfs = 4;
  auto bucket_id = hf0_(key) % num_buckets_;
  for (int hf = 0; hf < num_hfs; hf++) {
    auto pair = d_table_[bucket_id].load(cuda::memory_order_relaxed);
    INCREMENT_PROBES
    if (pair.first == key) {
      return pair.second;
    } else if (pair.first == sentinel_key_) {
      return sentinel_value_;
    } else {
      if (hf == 0) {
        bucket_id = hf1_(key) % num_buckets_;
      } else if (hf == 1) {
        bucket_id = hf2_(key) % num_buckets_;
      } else {
        bucket_id = hf3_(key) % num_buckets_;
      }
    }
  }

  return sentinel_value_;
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator>
template <typename RNG>
void bght::cht<Key, T, Hash, KeyEqual, Scope, Allocator>::randomize_hash_functions(
    RNG& rng) {
  hf0_ = initialize_hf<hasher>(rng);
  hf1_ = initialize_hf<hasher>(rng);
  hf2_ = initialize_hf<hasher>(rng);
  hf3_ = initialize_hf<hasher>(rng);
}
}  // namespace bght
