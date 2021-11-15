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
namespace bght {
namespace detail {
struct mars_rng_32 {
  uint32_t y;
  __host__ __device__ constexpr mars_rng_32() : y(2463534242) {}
  constexpr uint32_t __host__ __device__ operator()() {
    y ^= (y << 13);
    y = (y >> 17);
    return (y ^= (y << 5));
  }
};
}  // namespace detail
}  // namespace bght
