#pragma once
#include <cooperative_groups.h>
#include <detail/cuda_helpers.cuh>

namespace bght {
namespace detail {
namespace kernels {

template <typename InputIt, typename HashMap>
__global__ void insert_kernel(InputIt first, InputIt last, HashMap map) {
  // construct the tile
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size_>(block);

  auto count = last - first;
  if ((thread_id - tile.thread_rank()) >= count) {
    return;
  }

  bool do_op = false;
  typename HashMap::value_type insertion_pair{};

  // load the input
  if (thread_id < count) {
    insertion_pair = first[thread_id];
    do_op = true;
  }

  bool success = true;
  // Do the insertion
  auto work_queue = tile.ballot(do_op);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_pair = tile.shfl(insertion_pair, cur_rank);
    bool insertion_success = map.cooperative_insert(cur_pair, tile);

    if (tile.thread_rank() == cur_rank) {
      do_op = false;
      success = insertion_success;
    }
    work_queue = tile.ballot(do_op);
  }

  if (!tile.all(success)) {
    *map.d_build_success_ = false;
  }
}

template <typename InputIt, typename OutputIt, typename HashMap>
__global__ void find_kernel(InputIt first,
                            InputIt last,
                            OutputIt output_begin,
                            HashMap map) {
  // construct the tile
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size_>(block);

  auto count = last - first;
  if ((thread_id - tile.thread_rank()) >= count) {
    return;
  }

  bool do_op = false;
  typename HashMap::key_type find_key;
  typename HashMap::mapped_type result;

  // load the input
  if (thread_id < count) {
    find_key = first[thread_id];
    do_op = true;
  }

  // Do the insertion
  auto work_queue = tile.ballot(do_op);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key = tile.shfl(find_key, cur_rank);

    typename HashMap::mapped_type find_result = map.cooperative_find(cur_key, tile);

    if (tile.thread_rank() == cur_rank) {
      result = find_result;
      do_op = false;
    }
    work_queue = tile.ballot(do_op);
  }

  if (thread_id < count) {
    output_begin[thread_id] = result;
  }
}
}  // namespace kernels
}  // namespace detail
}  // namespace bght
