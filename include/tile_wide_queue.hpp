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

#include <cooperative_groups.h>
namespace bght {
template <typename T, typename CG>
struct tile_wide_queue {
  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = uint32_t;

  __device__ tile_wide_queue(const T& element, const T& sentinel, const CG& group)
      : element_{element}, cg_{group} {
    active_lane_ = (element_ != sentinel);
    build();
  }
  __device__ tile_wide_queue(const T& element, const bool& valid, const CG& group)
      : element_{element}, cg_{group} {
    active_lane_ = valid;
    build();
  }

  __device__ value_type front() {
    auto item = cg_.shfl(element_, cur_);
    return item;
  }
  __device__ void pop() {
    if (!empty()) {
      if (cg_.thread_rank() == cur_) {
        active_lane_ = false;
      }
      build();
    }
  }
  __device__ [[nodiscard]] bool empty() const { return mask_ == 0; }
  __device__ size_type size() const { return __popc(mask_); }

 private:
  __device__ void build() {
    mask_ = cg_.ballot(active_lane_);
    cur_ = __ffs(mask_) - 1;
  }
  const T& element_;
  const CG cg_;
  bool active_lane_;
  uint32_t mask_;
  uint32_t cur_;
};
}  // namespace bght
