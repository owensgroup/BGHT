#pragma once

#include <hip/hip_runtime.h>
#include <bght/pair.cuh>
#include <hip/std/utility>

namespace bght {
namespace detail {

template <typename T, typename tile_type>
__host__ __device__ static inline std::uint64_t ballot(
    const T value,
    [[maybe_unused]] const tile_type& tile) {
  using result_t = unsigned long long int;
  result_t result = __ballot64(value);
  return static_cast<std::uint64_t>(result);
}

template <typename T, typename tile_type>
__host__ __device__ static inline int any(const T value,
                                          [[maybe_unused]] const tile_type& tile) {
  using result_t = int;
  result_t result = __any(value);
  return static_cast<int>(result);
}

template <typename T, typename tile_type>
__host__ __device__ static inline int all(const T value,
                                          [[maybe_unused]] const tile_type& tile) {
  using result_t = int;
  result_t result = __all(value);
  return static_cast<int>(result);
}

template <typename T, typename tile_type>
__host__ __device__ static inline std::uint64_t popc(
    const T value,
    [[maybe_unused]] const tile_type& tile) {
  using result_t = unsigned int;
  result_t result = __popcll(value);
  return static_cast<std::uint64_t>(result);
}

template <typename T>
__host__ __device__ static inline std::uint64_t ffs(const T value) {
  using result_t = unsigned int;
  result_t result = __ffsll(static_cast<unsigned long long int>(value));
  return static_cast<std::uint64_t>(result);
}

template <typename T>
struct is_hip_std_pair : std::false_type {};

template <typename T1, typename T2>
struct is_hip_std_pair<hip::std::pair<T1, T2>> : std::true_type {};

template <typename T>
struct is_bght_pair : std::false_type {};

template <typename T1, typename T2>
struct is_bght_pair<bght::pair<T1, T2>> : std::true_type {};

template <typename T, typename tile_type>
__host__ __device__ static inline T shuffle(const T value,
                                            int src_lane,
                                            [[maybe_unused]] const tile_type& tile) {
  T result{};
  if constexpr (is_hip_std_pair<T>::value || is_bght_pair<T>::value) {
    result.first = tile.shfl(value.first, src_lane);
    result.second = tile.shfl(value.second, src_lane);
  } else {
    result = tile.shfl(value, src_lane);
  }

  return result;
}
}  // namespace detail
}  // namespace bght