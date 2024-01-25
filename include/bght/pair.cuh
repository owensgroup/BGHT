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
#include <bght/detail/pair_detail.hpp>
#include <type_traits>

namespace bght {
template <typename T1, typename T2, bool Padding = detail::padding_size<T1, T2>() != 0>
struct alignas(detail::pair_alignment<T1, T2>()) padded_pair {
  using first_type = T1;
  using second_type = T2;
  T1 first;
  T2 second;
  padded_pair() = default;
  ~padded_pair() = default;
  padded_pair(padded_pair const&) = default;
  padded_pair(padded_pair&&) = default;
  padded_pair& operator=(padded_pair const&) = default;
  padded_pair& operator=(padded_pair&&) = default;

  __host__ __device__ inline bool operator==(const padded_pair& rhs) const {
    return (this->first == rhs.first) && (this->second == rhs.second);
  }
  __host__ __device__ inline bool operator!=(const padded_pair& rhs) const {
    return !(*this == rhs);
  }

  __host__ __device__ constexpr padded_pair(T1 const& t, T2 const& u)
      : first{t}, second{u} {}
};

template <typename T1, typename T2>
struct alignas(detail::pair_alignment<T1, T2>()) padded_pair<T1, T2, true> {
  using first_type = T1;
  using second_type = T2;
  T1 first;
  T2 second;

  padded_pair() = default;
  ~padded_pair() = default;
  padded_pair(padded_pair const&) = default;
  padded_pair(padded_pair&&) = default;
  padded_pair& operator=(padded_pair const&) = default;
  padded_pair& operator=(padded_pair&&) = default;

  __host__ __device__ inline bool operator==(const padded_pair& rhs) const {
    return (this->first == rhs.first) && (this->second == rhs.second);
  }
  __host__ __device__ inline bool operator!=(const padded_pair& rhs) const {
    return !(*this == rhs);
  }

  __host__ __device__ constexpr padded_pair(T1 const& t, T2 const& u)
      : first{t}, second{u} {}

 private:
  char padding[detail::padding_size<T1, T2>()] = {0};
};

template <typename T1, typename T2>
using pair = padded_pair<T1, T2>;

template <class T = void>
struct equal_to {
  constexpr bool operator()(const T& lhs, const T& rhs) const { return lhs == rhs; }
};
}  // namespace bght
