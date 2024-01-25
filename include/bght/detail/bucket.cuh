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
#include <cuda/atomic>
#include <cuda/std/atomic>

namespace bght {
namespace detail {
template <typename atomic_pair_type, typename pair_type, typename tile_type>
struct bucket {
  bucket() = delete;
  DEVICE_QUALIFIER
  bucket(atomic_pair_type* ptr, const tile_type& tile) : ptr_(ptr), tile_(tile) {}

  DEVICE_QUALIFIER
  bucket(const bucket& other) : lane_pair_(other.lane_pair_), ptr_(other.ptr_) {}

  DEVICE_QUALIFIER
  void load(cuda::memory_order order = cuda::memory_order_seq_cst) {
    lane_pair_ = ptr_[tile_.thread_rank()].load(order);
  }
  DEVICE_QUALIFIER
  int compute_load(const pair_type& sentinel) {
    auto load_bitmap = tile_.ballot(lane_pair_.first != sentinel.first);
    int load = __popc(load_bitmap);
    return load;
  }
  // returns -1 if not found
  template <class KeyEqual>
  DEVICE_QUALIFIER int find_key_location(const typename pair_type::first_type& key,
                                         const KeyEqual key_equal) {
    bool key_exist = key_equal(key, lane_pair_.first);
    auto key_exist_bmap = tile_.ballot(key_exist);
    int key_lane = __ffs(key_exist_bmap);
    return key_lane - 1;
  }
  DEVICE_QUALIFIER
  typename pair_type::second_type get_value_from_lane(int location) {
    return tile_.shfl(lane_pair_.second, location);
  }

  DEVICE_QUALIFIER
  bool weak_cas_at_location(const pair_type& pair,
                            const int location,
                            const pair_type& sentinel,
                            cuda::memory_order success = cuda::memory_order_seq_cst,
                            cuda::memory_order failure = cuda::memory_order_seq_cst) {
    pair_type expected = sentinel;
    pair_type desired = pair;
    bool cas_success =
        ptr_[location].compare_exchange_weak(expected, desired, success, failure);
    return cas_success;
  }

  DEVICE_QUALIFIER
  bool strong_cas_at_location(const pair_type& pair,
                              const int location,
                              const pair_type& sentinel,
                              cuda::memory_order success = cuda::memory_order_seq_cst,
                              cuda::memory_order failure = cuda::memory_order_seq_cst) {
    pair_type expected = sentinel;
    pair_type desired = pair;
    bool cas_success =
        ptr_[location].compare_exchange_strong(expected, desired, success, failure);
    return cas_success;
  }

  DEVICE_QUALIFIER
  pair_type exch_at_location(const pair_type& pair,
                             const int location,
                             cuda::memory_order order = cuda::memory_order_seq_cst) {
    auto old = ptr_[location].exchange(pair, order);
    return old;
  }

 private:
  pair_type lane_pair_;
  atomic_pair_type* ptr_;
  tile_type tile_;
};
}  // namespace detail
}  // namespace bght
