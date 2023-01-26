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

#include <hip/hip_cooperative_groups.h>

namespace bght {
namespace detail {
namespace groups {

// __device__ auto this_thread_block() {
//   return cooperative_groups::this_thread_block();
// }
// template <unsigned int Size, typename ParentT>
// struct partition
//     : public cooperative_groups::impl::tiled_partition_internal<Size, ParentT> {
//   __device__ partition(const ParentT& parent)
//       : cooperative_groups::impl::tiled_partition_internal<Size, ParentT>(parent) {}
//   __device__ static uint64_t ballot(int predicate) { return __ballot(predicate); }
//   template <typename T>
//   __device__ auto shfl(T var, int src_lane) const {
//     static_assert(sizeof(T) <= 8);
//     if constexpr (sizeof(T) > 4) {
//       T result;
//       result.first =
//           cooperative_groups::impl::tiled_partition_internal<Size, ParentT>::shfl(
//               var.first, src_lane);
//       result.second =
//           cooperative_groups::impl::tiled_partition_internal<Size, ParentT>::shfl(
//               var.second, src_lane);
//       return result;
//     } else {
//       return cooperative_groups::impl::tiled_partition_internal<Size, ParentT>::shfl(
//           var, src_lane);
//     }
//   }
//   __device__ auto all(int predicate) const { return __all(predicate); }
// };
};  // namespace groups

}  // namespace detail

}  // namespace bght
