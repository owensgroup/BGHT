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

#include "hip_helpers.hpp"

#ifdef COUNT_PROBES
__device__ __managed__ uint32_t global_probes_count = 0;
#define INCREMENT_PROBES_IN_TILE \
  if (tile.thread_rank() == 0)   \
    atomicAdd(&global_probes_count, 1);
#define INCREMENT_PROBES atomicAdd(&global_probes_count, 1);
namespace bght {
inline uint32_t get_num_probes() {
  hip_try(hipDeviceSynchronize());
  auto count = global_probes_count;
  global_probes_count = 0;
  hip_try(hipDeviceSynchronize());
  return count;
}
}  // namespace bght
#else
#define INCREMENT_PROBES_IN_TILE
#define INCREMENT_PROBES

#endif
