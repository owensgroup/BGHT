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

namespace bght {
namespace detail {
namespace bits {
// Bit Field Extract.
__device__ __forceinline__ int bfe(uint32_t src, int num_bits) {
  unsigned mask;
  asm("bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits));
  return mask;
}

// Find most significant non - sign bit.
// bfind(0) = -1, bfind(1) = 0
__device__ __forceinline__ int bfind(uint32_t src) {
  int msb;
  asm("bfind.u32 %0, %1;" : "=r"(msb) : "r"(src));
  return msb;
}
__device__ __forceinline__ int bfind(uint64_t src) {
  int msb;
  asm("bfind.u64 %0, %1;" : "=r"(msb) : "l"(src));
  return msb;
}
};  // namespace bits
}  // namespace detail
}  // namespace bght
