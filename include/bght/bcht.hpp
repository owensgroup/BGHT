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
#include <bght/allocator.hpp>
#include <bght/detail/cuda_helpers.cuh>
#include <bght/detail/kernels.cuh>
#include <bght/hash_functions.hpp>
#include <bght/pair.cuh>
#include <cuda/atomic>
#include <cuda/std/utility>
#include <memory>

namespace bght {

/**
 * @brief BCHT BCHT (bucketed cuckoo hash table) is an associative static GPU hash table
 * that contains key-value pairs with unique keys. The hash table is an open addressing
 * hash table based on the cuckoo hashing probing scheme (bucketed and using three hash
 * functions).
 *
 * @tparam Key Type for the hash map key
 * @tparam T Type for the mapped value
 * @tparam Hash Unary function object class that defines the hash function. The function
 * must have an `initialize_hf` specialization to initialize the hash function using a
 * random number generator
 * @tparam KeyEqual Binary function object class that compares two keys
 * @tparam Allocator The allocator to use for allocating GPU device memory
 * @tparam B Bucket size for the hash table
 */
template <class Key,
          class T,
          class Hash = bght::MurmurHash3_32<Key>,
          class KeyEqual = bght::equal_to<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator = bght::cuda_allocator<char>,
          int B = 16>
struct bcht {
  using value_type = pair<Key, T>;
  using key_type = Key;
  using mapped_type = T;
  using atomic_pair_type = cuda::atomic<value_type, Scope>;
  using allocator_type = Allocator;
  using hasher = Hash;
  using size_type = std::size_t;

  using atomic_pair_allocator_type =
      typename std::allocator_traits<Allocator>::rebind_alloc<atomic_pair_type>;
  using pool_allocator_type =
      typename std::allocator_traits<Allocator>::rebind_alloc<bool>;
  using size_type_allocator_type =
      typename std::allocator_traits<Allocator>::rebind_alloc<size_type>;

  static constexpr auto bucket_size = B;
  using key_equal = KeyEqual;

  /**
   * @brief Constructs the hash table with the specified capacity and uses the specified
   * sentinel key and value to define a sentinel pair.
   *
   * @param capacity The number of slots to use in the hash table. If the capacity is not
   * multiple of the bucket size, it will be rounded
   * @param sentinel_key A reserved sentinel key that defines an empty key
   * @param sentinel_value A reserved sentinel value that defines an empty value
   * @param allocator The allocator to use for allocating GPU device memory
   */
  bcht(std::size_t capacity,
       Key sentinel_key,
       T sentinel_value,
       Allocator const& allocator = Allocator{});
  /**
   * @brief A shallow-copy constructor
   */
  bcht(const bcht& other);
  /**
   * @brief Move constructor is currently deleted
   */
  bcht(bcht&&) = delete;
  /**
   * @brief The assignment operator for the BCHT is currently deleted
   */
  bcht& operator=(const bcht&) = delete;
  /**
   * @brief The move assignment operator for the BCHT is currently deleted
   */
  bcht& operator=(bcht&&) = delete;
  /**
   * @brief Destructor that destroys the hash map and deallocate memory if no copies exist
   */
  ~bcht();

  /**
   * @brief Clears the hash map and resets all slots
   */
  void clear();

  /**
   * @brief Host-side API for inserting all pairs defined by the input argument iterators.
   * All keys in the range must be unique and must not exist in the hash table.
   * @tparam InputIt Device-side iterator that can be converted to `value_type`.
   * @param first An iterator defining the beginning of the input pairs to insert
   * @param last  An iterator defining the end of the input pairs to insert
   * @param stream  A CUDA stream where the insertion operation will take place
   * @return A boolean indicating success (true) or failure (false) of the insertion
   * operation.
   */
  template <typename InputIt>
  bool insert(InputIt first, InputIt last, cudaStream_t stream = 0);

  /**
   * @brief Host-side API for finding all keys defined by the input argument iterators.
   * @tparam InputIt  Device-side iterator that can be converted to `key_type`
   * @tparam OutputIt Device-side iterator that can be converted to `mapped_type`
   * @param first An iterator defining the beginning of the input keys to find
   * @param last An iterator defining the end of the input keys to find
   * @param output_begin An iterator defining the beginning of the output buffer to store
   * the results into. The size of the buffer must match the number of queries defined by
   * the input iterators.
   * @param stream  A CUDA stream where the insertion operation will take place
   */
  template <typename InputIt, typename OutputIt>
  void find(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0);

  /**
   * @brief Device-side cooperative insertion API that inserts a single pair into the hash
   * map.
   * @tparam tile_type A cooperative group tile with a size that must match the bucket
   * size of the hash map (i.e., `bucket_size`). It must support the tile-wide intrinsics
   * `ballot`, `shfl`
   * @param pair A key-value pair to insert into the hash map. The pair must be the same
   * for all threads in the  cooperative group tile
   * @param tile  The cooperative group tile
   * @return A boolean indicating success (true) or failure (false) of the insertion
   * operation.
   */
  template <typename tile_type>
  __device__ bool insert(value_type const& pair, tile_type const& tile);

  /**
   * @brief Device-side cooperative find API that finds a single pair into the hash
   * map.
   * @tparam tile_type A cooperative group tile with a size that must match the bucket
   * size of the hash map (i.e., `bucket_size`). It must support the tile-wide intrinsics
   * `ballot`, `shfl`
   * @param key A key to find in the hash map. The key must be the same
   * for all threads in the  cooperative group tile
   * @param tile The cooperative group tile
   * @return The value of the key if it exists in the map or the `sentinel_value` if the
   * key does not exist in the hash map
   */
  template <typename tile_type>
  __device__ mapped_type find(key_type const& key, tile_type const& tile);

  /**
   * @brief Host-side API to randomize the hash functions used for the probing scheme.
   * This can be used when the hash table construction fails. The hash table must be
   * cleared after a call to this function.
   * @tparam RNG A pseudo-random number generator
   * @param rng An instantiation of the pseudo-random number generator
   */
  template <typename RNG>
  void randomize_hash_functions(RNG& rng);

  /**
   * @brief Compute the number of elements in the map
   * @return The number of elements in the map
   */
  size_type size(cudaStream_t stream = 0);

 private:
  template <typename InputIt, typename HashMap>
  friend __global__ void detail::kernels::tiled_insert_kernel(InputIt, InputIt, HashMap);

  template <typename InputIt, typename OutputIt, typename HashMap>
  friend __global__ void detail::kernels::tiled_find_kernel(InputIt,
                                                            InputIt,
                                                            OutputIt,
                                                            HashMap);

  template <int BlockSize, typename InputT, typename HashMap>
  friend __global__ void detail::kernels::count_kernel(const InputT,
                                                       std::size_t*,
                                                       HashMap);

  std::size_t capacity_;
  key_type sentinel_key_{};
  mapped_type sentinel_value_{};
  allocator_type allocator_;
  atomic_pair_allocator_type atomic_pairs_allocator_;
  pool_allocator_type pool_allocator_;
  size_type_allocator_type size_type_allocator_;

  atomic_pair_type* d_table_{};
  std::shared_ptr<atomic_pair_type> table_;

  bool* d_build_success_;
  std::shared_ptr<bool> build_success_;

  uint32_t max_cuckoo_chains_;

  Hash hf0_;
  Hash hf1_;
  Hash hf2_;

  std::size_t num_buckets_;
};

template <typename Key, typename T>
using bcht8 = typename bght::bcht<Key,
                                  T,
                                  bght::MurmurHash3_32<Key>,
                                  bght::equal_to<Key>,
                                  cuda::thread_scope_device,
                                  bght::cuda_allocator<char>,
                                  8>;

template <typename Key, typename T>
using bcht16 = typename bght::bcht<Key,
                                   T,
                                   bght::MurmurHash3_32<Key>,
                                   bght::equal_to<Key>,
                                   cuda::thread_scope_device,
                                   bght::cuda_allocator<char>,
                                   16>;

template <typename Key, typename T>
using bcht32 = typename bght::bcht<Key,
                                   T,
                                   bght::MurmurHash3_32<Key>,
                                   bght::equal_to<Key>,
                                   cuda::thread_scope_device,
                                   bght::cuda_allocator<char>,
                                   32>;

}  // namespace bght

#include <bght/detail/bcht_impl.cuh>
