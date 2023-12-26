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
template <typename Key>
struct universal_hash {
  using key_type = Key;
  using result_type = Key;

  __host__ __device__ constexpr universal_hash() : hash_x_(1), hash_y_(2) {}

  __host__ __device__ constexpr universal_hash(uint32_t hash_x, uint32_t hash_y)
      : hash_x_(hash_x), hash_y_(hash_y) {}

  constexpr result_type __host__ __device__ operator()(const key_type& key) const {
    return (((hash_x_ ^ key) + hash_y_) % prime_divisor);
  }

  universal_hash(const universal_hash&) = default;
  universal_hash(universal_hash&&) = default;
  universal_hash& operator=(universal_hash const&) = default;
  universal_hash& operator=(universal_hash&&) = default;
  ~universal_hash() = default;

  static constexpr uint32_t prime_divisor = 4294967291u;

 private:
  uint32_t hash_x_;
  uint32_t hash_y_;
};
}  // namespace bght