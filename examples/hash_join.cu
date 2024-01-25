

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <bght/iht.hpp>
#include <bght/tile_wide_queue.hpp>
#include <limits>
#include <type_traits>
#include <vector>

template <typename HashMap>
__global__ void join(HashMap read_from, HashMap query_into, HashMap result_table) {
  using pair_type = typename HashMap::value_type;
  using key_type = typename HashMap::key_type;
  using value_type = typename HashMap::mapped_type;

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  using queue_type = bght::tile_wide_queue<pair_type, decltype(tile)>;

  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  const auto capacity = read_from.max_size();
  const auto begin = read_from.begin();

  const auto sentinel_pair = read_from.get_sentinel_pair();

  const auto pair = thread_id < capacity
                        ? (begin + thread_id)->load(cuda::memory_order_relaxed)
                        : sentinel_pair;

  queue_type work_queue(pair, sentinel_pair, tile);

  while (!work_queue.empty()) {
    auto cur_pair = work_queue.front();
    work_queue.pop();

    auto value = query_into.find(cur_pair.first, tile);
    // if the key exist
    if (value != sentinel_pair.second) {
      // store into the result table the sum of the two values
      pair_type new_pair{cur_pair.first, cur_pair.second + value};
      result_table.insert(new_pair, tile);
    }
  }
}

template <typename HashMap>
__global__ void print(HashMap result_table) {
  using pair_type = typename HashMap::value_type;
  using key_type = typename HashMap::key_type;
  using value_type = typename HashMap::mapped_type;

  const auto capacity = result_table.max_size();
  const auto sentinel_key = result_table.get_sentinel_key();
  const auto sentinel_pair = result_table.get_sentinel_pair();

  auto begin = result_table.begin();

  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  const auto pair = thread_id < capacity
                        ? (begin + thread_id)->load(cuda::memory_order_relaxed)
                        : sentinel_pair;
  if (pair.first != sentinel_key) {
    printf("result_map[%u] = %u, exepcted %u\n",
           pair.first,
           pair.second,
           pair.first * 2 * 10);
  };
}
int main(int, char**) {
  using key_type = uint32_t;
  using value_type = uint32_t;
  using pair_type = bght::pair<key_type, value_type>;

  std::vector<pair_type> read_from_input_h = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}};
  std::vector<pair_type> query_into_input_h = {
      {3, 30}, {4, 40}, {5, 50}, {6, 60}, {7, 70}};

  auto invalid_key = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<value_type>::max();

  const float load_factor = 0.7;

  const auto num_read_from_keys = read_from_input_h.size();
  const auto num_query_into_keys = query_into_input_h.size();

  std::size_t read_from_capacity = double(num_read_from_keys) / load_factor;
  std::size_t query_into_capacity = double(num_query_into_keys) / load_factor;

  using hash_map = bght::iht<key_type, value_type>;

  hash_map read_from_map(read_from_capacity, invalid_key, invalid_value);
  hash_map query_into_map(query_into_capacity, invalid_key, invalid_value);

  thrust::device_vector<pair_type> read_from_input(read_from_input_h);
  thrust::device_vector<pair_type> query_into_input(query_into_input_h);

  // do the insertions which can execute concurrently, but here we use the default stream
  bool success = read_from_map.insert(read_from_input.data().get(),
                                      read_from_input.data().get() + num_read_from_keys);
  if (!success) {
    std::cerr << "Failed to build first map" << std::endl;
  }
  success = query_into_map.insert(query_into_input.data().get(),
                                  query_into_input.data().get() + query_into_capacity);
  if (!success) {
    std::cerr << "Failed to build second map" << std::endl;
  }

  // build the result
  const auto worst_case_capacity = std::min(query_into_capacity, read_from_capacity);
  hash_map result_map(worst_case_capacity, invalid_key, invalid_value);

  const uint32_t block_size = 128;
  uint32_t num_blocks = (num_read_from_keys + block_size - 1) / block_size;
  join<<<num_blocks, block_size>>>(read_from_map, query_into_map, result_map);

  cuda_try(cudaDeviceSynchronize());

  std::cout << "Join complete" << std::endl;

  num_blocks = (worst_case_capacity + block_size - 1) / block_size;
  print<<<num_blocks, block_size>>>(result_map);
}