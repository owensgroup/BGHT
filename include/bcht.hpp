#pragma once
#include <cuda/atomic>
#include <cuda/std/utility>
#include <detail/allocator.hpp>
#include <detail/cuda_helpers.cuh>
#include <detail/hash_functions.cuh>
#include <detail/kernels.cuh>
#include <detail/pair.cuh>
#include <memory>

namespace bght {
template <class Key,
          class T,
          class Hash = bght::detail::universal_hash<Key>,
          class KeyEqual = bght::equal_to<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator = bght::cuda_allocator<char>,
          int B = 16>
struct bcht {
  using value_type = pair<Key, T>;
  // using value_type = cuda::std::pair<const Key, T>;
  using key_type = Key;
  using mapped_type = T;
  using atomic_pair_type = cuda::atomic<value_type, Scope>;
  using allocator_type = Allocator;
  using hasher = Hash;
  using atomic_pair_allocator_type =
      typename std::allocator_traits<Allocator>::rebind_alloc<atomic_pair_type>;
  using bool_allocator_type =
      typename std::allocator_traits<Allocator>::rebind_alloc<bool>;

  bcht(std::size_t capacity,
       Key sentinel_key,
       T sentinel_value,
       Allocator const& allocator = Allocator{});
  bcht(const bcht& other);
  bcht(bcht&&) = delete;
  bcht& operator=(const bcht&) = delete;
  bcht& operator=(bcht&&) = delete;
  ~bcht();

  template <typename InputIt>
  bool insert(InputIt first, InputIt last, cudaStream_t stream = 0);

  template <typename InputIt, typename OutputIt>
  void find(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0);

  template <typename tile_type>
  __device__ bool cooperative_insert(value_type const& pair, tile_type const& tile);

  template <typename tile_type>
  __device__ mapped_type cooperative_find(key_type const& key, tile_type const& tile);

 private:
  template <typename InputIt, typename HashMap>
  friend __global__ void detail::kernels::insert_kernel(InputIt, InputIt, HashMap);

  template <typename InputIt, typename OutputIt, typename HashMap>
  friend __global__ void detail::kernels::find_kernel(InputIt,
                                                      InputIt,
                                                      OutputIt,
                                                      HashMap);

  static constexpr auto bucket_size_ = B;

  std::size_t capacity_;
  key_type sentinel_key_{};
  mapped_type sentinel_value_{};
  allocator_type allocator_;
  atomic_pair_allocator_type atomic_pairs_allocator_;
  bool_allocator_type bool_allocator_;

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
}  // namespace bght

#include <detail/bcht_impl.cuh>
