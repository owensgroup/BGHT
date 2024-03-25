/*
 *   Copyright 2024 The Regents of the University of California, Davis
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

#include <cuda/std/utility>
#include <type_traits>

namespace bght {
namespace detail {

template <typename T>
struct is_cuda_std_pair : std::false_type {};

template <typename T1, typename T2>
struct is_cuda_std_pair<cuda::std::pair<T1, T2>> : std::true_type {};

template <typename T, typename Tile>
__device__ T shuffle(const T& value, int location, const Tile& tile) {
  T result{};
  if constexpr (is_cuda_std_pair<T>::value) {
    result.first = tile.shfl(value.first, location);
    result.second = tile.shfl(value.second, location);
  } else {
    result = tile.shfl(value, location);
  }

  return result;
}
}  // namespace detail
}  // namespace bght