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
#include <type_traits>
namespace bght {
namespace detail {
template <typename T>
constexpr std::size_t next_alignment() {
  constexpr std::size_t n = sizeof(T);
  if (n <= 4)
    return 4;
  if (n <= 8)
    return 8;
  return 16;
}
constexpr std::size_t next_alignment(std::size_t n) {
  if (n <= 4)
    return 4;
  if (n <= 8)
    return 8;
  return 16;
}

template <typename T1, typename T2>
constexpr std::size_t pair_size() {
  return sizeof(T1) + sizeof(T2);
}

template <typename T1, typename T2>
constexpr std::size_t pair_alignment() {
  return next_alignment(pair_size<T1, T2>());
}

template <typename T1, typename T2>
constexpr std::size_t padding_size() {
  constexpr auto psz = pair_size<T1, T2>();
  constexpr auto apsz = next_alignment(pair_size<T1, T2>());
  if (psz > apsz) {
    constexpr auto nsz = (1ull + (psz / apsz)) * apsz;
    return nsz - psz;
  }
  return apsz - psz;
}
}  // namespace detail
}  // namespace bght
