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

#include <bght/detail/MurmurHash3_32.hpp>
#include <bght/detail/universal_hash.hpp>

namespace bght {

template <typename Hash, typename RNG>
Hash initialize_hf(RNG& rng) {
  if constexpr (std::is_same_v<Hash, universal_hash<typename Hash::key_type>>) {
    uint32_t x = rng() % Hash::prime_divisor;
    if (x < 1u) {
      x = 1;
    }
    uint32_t y = rng() % Hash::prime_divisor;
    return Hash(x, y);
  }

  if constexpr (std::is_same_v<Hash, MurmurHash3_32<typename Hash::key_type>>) {
    uint32_t x = rng();
    if (x < 1u) {
      x = 1;
    }
    return Hash(x);
  }
  return Hash{};
}

}  // namespace bght
