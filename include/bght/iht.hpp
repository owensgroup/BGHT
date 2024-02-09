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
#include <bght/allocator.hpp>
#include <bght/detail/cuda_helpers.cuh>
#include <bght/detail/kernels.cuh>
#include <bght/detail/prime.hpp>
#include <bght/hash_functions.hpp>
#include <bght/pair.cuh>
#include <cuda/atomic>
#include <cuda/std/utility>
#include <memory>

namespace bght {

/**
 * @brief IHT IHT (iceberg hash table) is an associative static GPU hash table
 * that contains key-value pairs with unique keys. The hash table is an open addressing
 * hash table based on the double hashing probing scheme (bucketed and using a primary
 * hash function followed by double hashing).
 *
 * @tparam Key Type for the hash map key
 * @tparam T Type for the mapped value
 * @tparam Hash Unary function object class that defines the hash function. The function
 * must have an `initialize_hf` specialization to initialize the hash function using a
 * random number generator
 * @tparam KeyEqual Binary function object class that compares two keys
 * @tparam Allocator The allocator to use for allocating GPU device memory
 * @tparam B Bucket size for the hash table
 * @tparam Threshold Iceberg threshold
 */
template <class Key,
          class T,
          class Hash = bght::MurmurHash3_32<Key>,
          class KeyEqual = bght::equal_to<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator = bght::cuda_allocator<char>,
          int B = 16,
          int Threshold = 14>
struct iht {
  static_assert(Threshold < B, "Threshold must be less than the bucket size");

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

  using iterator = atomic_pair_type*;
  using const_iterator = iterator;

  /**
   * @brief Constructs the hash table with the specified capacity and uses the specified
   * sentinel key and value to define a sentinel pair.
   *
   * @param capacity The number of slots to use in the hash table. If the capacity is
   * not multiple of the bucket size, it will be rounded
   * @param sentinel_key A reserved sentinel key that defines an empty key
   * @param sentinel_value A reserved sentinel value that defines an empty value
   * @param allocator The allocator to use for allocating GPU device memory
   */
  iht(std::size_t capacity,
      Key sentinel_key,
      T sentinel_value,
      Allocator const& allocator = Allocator{});
  /**
   * @brief A shallow-copy constructor
   */
  iht(const iht& other);
  /**
   * @brief Move constructor is currently deleted
   */
  iht(iht&&) = delete;
  /**
   * @brief The assignment operator is currently deleted
   */
  iht& operator=(const iht&) = delete;
  /**
   * @brief The move assignment operator is currently deleted
   */
  iht& operator=(iht&&) = delete;
  /**
   * @brief Destructor that destroys the hash map and deallocate memory if no copies
   * exist
   */
  ~iht();
  /**
   * @brief Clears the hash map and resets all slots
   */
  void clear();

  /**
   * @brief Host-side API for inserting all pairs defined by the input argument
   * iterators. All keys in the range must be unique and must not exist in the hash
   * table.
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
   * @param output_begin An iterator defining the beginning of the output buffer to
   * store the results into. The size of the buffer must match the number of queries
   * defined by the input iterators.
   * @param stream  A CUDA stream where the insertion operation will take place
   */
  template <typename InputIt, typename OutputIt>
  void find(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0);

  /**
   * @brief Device-side cooperative insertion API that inserts a single pair into the
   * hash map if the key does not exist.
   * @tparam tile_type A cooperative group tile with a size that must match the bucket
   * size of the hash map (i.e., `bucket_size`). It must support the tile-wide
   * intrinsics `ballot`, `shfl`
   * @param pair A key-value pair to insert into the hash map. The pair must be the same
   * for all threads in the  cooperative group tile
   * @param tile  The cooperative group tile
   * @return A pair where the second element is a boolean indicating success (true)
   * or failure (false) of the insertion operation. If insertion succeeded or the key
   * exists, the first element in the pair contain a pointer to the inserted or old
   * key-value pair, otherwise, the first pair element contain a pointer to the end of the
   * map.
   */
  template <typename tile_type>
  __device__ cuda::std::pair<iterator, bool> insert(value_type const& pair,
                                                    tile_type const& tile);

  /**
   * @brief Device-side cooperative find API that finds a single pair into the hash
   * map.
   * @tparam tile_type A cooperative group tile with a size that must match the bucket
   * size of the hash map (i.e., `bucket_size`). It must support the tile-wide
   * intrinsics `ballot`, `shfl`
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

  /**
   * @brief Returns an iterator to the first element of the tables including all invalid
   * entries.
   *
   * @return const_iterator constant iterator to the first element of the table
   */
  __device__ __host__ const_iterator begin() const;
  /**
   * @brief Returns an iterator to the last element of the tables including all invalid
   * entries.
   *
   * @return const_iterator constant iterator to the last element of the table
   */
  __device__ __host__ const_iterator end() const;

  /**
   * @brief Returns an iterator to the first element of the tables including all invalid
   * entries.
   *
   * @return iterator constant iterator to the first element of the table
   */
  __device__ __host__ iterator begin();
  /**
   * @brief Returns an iterator to the last element of the tables including all invalid
   * entries.
   *
   * @return iterator constant iterator to the last element of the table
   */
  __device__ __host__ iterator end();

  /**
   * @brief Returns the maximum number of elements the container is able to hold
   *
   * @return size_type maximum number of elements including all invalid entries.
   */
  __device__ __host__ size_type max_size() const;

  /**
   * @brief Get the sentinel key object
   *
   * @return key_type Sentinel key
   */
  __device__ __host__ key_type get_sentinel_key() const { return sentinel_key_; }

  /**
   * @brief Get the sentinel value object
   *
   * @return mapped_type Sentinel value
   */
  __device__ __host__ mapped_type get_sentinel_value() const { return sentinel_value_; }

  /**
   * @brief Get the sentinel pair object
   *
   * @return value_type Sentinel pair
   */
  __device__ __host__ value_type get_sentinel_pair() const {
    return {get_sentinel_key(), get_sentinel_value()};
  }

  /**
   * @brief Checks if the hash map is empty
   *
   * @param stream A stream to enqueue the kernels on.

   * @return A boolean with value `true` if the container is empty or `false` if the
   * container is empty.
   */
  [[nodiscard]] bool empty(cudaStream_t stream = 0);

  /**
   * @brief Checks if a key exists in the hash map
   *
   * @tparam tile_type A cooperative group tile with a size that must match the bucket
   * size of the hash map (i.e., `bucket_size`). It must support the tile-wide
   * intrinsics `ballot`, `shfl`
   * @param key A key to find in the hash map. The key must be the same
   * for all threads in the  cooperative group tile
   * @param tile The cooperative group tile
   * @return The value of the key if it exists in the map or the `sentinel_value` if the
   * key does not exist in the hash map
   * @ return A boolean indicating whether the key exist or not .
   * */
  template <typename tile_type>
  __device__ bool contains(const Key& key, tile_type const& tile);

 private:
  template <typename InputIt, typename HashMap>
  friend __global__ void detail::kernels::tiled_insert_kernel(InputIt, InputIt, HashMap);

  template <typename InputIt, typename HashMap>
  friend __global__ void detail::kernels::iht_tiled_insert_kernel(InputIt,
                                                                  InputIt,
                                                                  HashMap);

  template <typename InputIt, typename OutputIt, typename HashMap>
  friend __global__ void detail::kernels::tiled_find_kernel(InputIt,
                                                            InputIt,
                                                            OutputIt,
                                                            HashMap);

  template <int BlockSize, typename InputT, typename HashMap>
  friend __global__ void detail::kernels::count_kernel(const InputT,
                                                       std::size_t*,
                                                       HashMap);

  static constexpr auto threshold_ = Threshold;

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

  Hash hfp_;
  Hash hf0_;
  Hash hf1_;

  std::size_t num_buckets_;
};

template <typename Key, typename T, int Threshold = 6>
using iht8 = typename bght::iht<Key,
                                T,
                                bght::MurmurHash3_32<Key>,
                                bght::equal_to<Key>,
                                cuda::thread_scope_device,
                                bght::cuda_allocator<char>,
                                8,
                                Threshold>;

template <typename Key, typename T, int Threshold = 12>
using iht16 = typename bght::iht<Key,
                                 T,
                                 bght::MurmurHash3_32<Key>,
                                 bght::equal_to<Key>,
                                 cuda::thread_scope_device,
                                 bght::cuda_allocator<char>,
                                 16,
                                 Threshold>;
template <typename Key, typename T, int Threshold = 25>
using iht32 = typename bght::iht<Key,
                                 T,
                                 bght::MurmurHash3_32<Key>,
                                 bght::equal_to<Key>,
                                 cuda::thread_scope_device,
                                 bght::cuda_allocator<char>,
                                 32,
                                 Threshold>;

}  // namespace bght

#include <bght/detail/iht_impl.cuh>
