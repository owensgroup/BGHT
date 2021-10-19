
#pragma once
#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <bcht.hpp>
#include <detail/bucket.cuh>
#include <detail/kernels.cuh>
#include <detail/rng.hpp>
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
    , bool_allocator_{allocator} {
  // capacity_ must be multiple of bucket size
  auto remainder = capacity_ % bucket_size_;
  if (remainder) {
    capacity_ += (bucket_size_ - remainder);
  }
  num_buckets_ = capacity_ / bucket_size_;

  d_table_ = std::allocator_traits<atomic_pair_allocator_type>::allocate(
      atomic_pairs_allocator_, capacity_);
  table_ =
      std::shared_ptr<atomic_pair_type>(d_table_, bght::cuda_deleter<atomic_pair_type>());

  d_build_success_ =
      std::allocator_traits<bool_allocator_type>::allocate(bool_allocator_, 1);
  build_success_ = std::shared_ptr<bool>(d_build_success_, bght::cuda_deleter<bool>());

  value_type empty_pair{sentinel_key_, sentinel_value_};

  thrust::fill(thrust::device, d_table_, d_table_ + capacity_, empty_pair);

  // maximum number of cuckoo chains
  double lg_input_size = (float)(log((double)capacity) / log(2.0));
  const unsigned max_iter_const = 7;
  max_cuckoo_chains_ = static_cast<uint32_t>(max_iter_const * lg_input_size);

  std::mt19937 rng(2);
  hf0_ = detail::initialize_hf<hasher>(rng);
  hf1_ = detail::initialize_hf<hasher>(rng);
  hf2_ = detail::initialize_hf<hasher>(rng);

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
    , bool_allocator_(other.bool_allocator_)
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
template <typename InputIt>
bool bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::insert(InputIt first,
                                                               InputIt last,
                                                               cudaStream_t stream) {
  const auto num_keys = last - first;
  const uint32_t block_size = 128;
  const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
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

  detail::kernels::find_kernel<<<num_blocks, block_size, 0, stream>>>(
      first, last, output_begin, *this);
  // cuda_try(cudaPeekAtLastError());
}

template <class Key,
          class T,
          class Hash,
          class KeyEqual,
          cuda::thread_scope Scope,
          class Allocator,
          int B>
template <typename tile_type>
__device__ bool
bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::cooperative_insert(
    value_type const& pair,
    tile_type const& tile) {
  detail::mars_rng_32 rng;
  // if (2514071929 == pair.first) {
  //  printf("Error\n");
  //}
  auto bucket_id = hf0_(pair.first) % num_buckets_;
  uint32_t cuckoo_counter = 0;
  auto lane_id = tile.thread_rank();
  const int elected_lane = 0;
  value_type sentinel_pair{sentinel_key_, sentinel_value_};
  value_type insertion_pair = pair;
  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;
  do {
    bucket_type cur_bucket(&d_table_[bucket_id * bucket_size_], tile);
    cur_bucket.load(cuda::memory_order_relaxed);

    int load = cur_bucket.compute_load(sentinel_pair);

    if (load != bucket_size_) {
      // bucket is not full
      bool cas_success = false;
      if (lane_id == elected_lane) {
        cas_success = cur_bucket.weak_cas_at_location(insertion_pair,
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
        auto random_location = rng() % bucket_size_;
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
  printf("failed to insert %i\n", pair.first);
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
bght::bcht<Key, T, Hash, KeyEqual, Scope, Allocator, B>::cooperative_find(
    key_type const& key,
    tile_type const& tile) {
  const int num_hfs = 3;
  auto bucket_id = hf0_(key) % num_buckets_;
  using bucket_type = detail::bucket<atomic_pair_type, value_type, tile_type>;
  for (int hf = 0; hf < num_hfs; hf++) {
    bucket_type cur_bucket(&d_table_[bucket_id * bucket_size_], tile);
    cur_bucket.load(cuda::memory_order_relaxed);
    int key_location = cur_bucket.find_key_location(key);
    if (key_location != -1) {
      auto found_value = cur_bucket.get_value_from_lane(key_location);
      return found_value;
    } else {
      bucket_id = hf == 0 ? hf1_(key) % num_buckets_ : hf2_(key) % num_buckets_;
    }
  }

  return sentinel_value_;
}
}  // namespace bght
