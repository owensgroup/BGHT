#include <thrust/device_vector.h>
#include <cuda/std/array>

#include <cstdint>

#include <bght/bcht.hpp>

// Testing passing a hashmap to the device
template <typename HashMap, typename Keys>
__global__ void test_kernel(HashMap map, Keys* keys) {
  using pair_type = typename HashMap::value_type;
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  // tile
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  // pair to insert
  auto key_id = thread_id / HashMap::bucket_size;
  const auto key = keys[key_id];
  const auto value = static_cast<typename HashMap::mapped_type>(key[0] * 10);

  if (tile.thread_rank() == 0) {
    printf("inserting keys[%i] = %i, value %i\n", key_id, key[0], value);
  }

  pair_type pair{key, value};

  map.insert(pair, tile);

  auto find_result = map.find(pair.first, tile);

  if (tile.thread_rank() == 0) {
    printf("found value for keys[%i] is %i, expected %i\n",
           key_id,
           find_result,
           pair.second);
  }
}
template <typename Key>
struct array_hash {
  using key_type = Key;
  using result_type = std::size_t;
  constexpr array_hash(uint32_t hash_x, uint32_t hash_y)
      : hash_x_(hash_x), hash_y_(hash_y) {}

  // just hash the first entry
  constexpr result_type __host__ __device__ operator()(const key_type& key) const {
    return (((hash_x_ ^ key[0]) + hash_y_) % prime_divisor);
  }

  array_hash(const array_hash&) = default;
  array_hash() = default;
  array_hash(array_hash&&) = default;
  array_hash& operator=(array_hash const&) = default;
  array_hash& operator=(array_hash&&) = default;
  ~array_hash() = default;
  static constexpr uint32_t prime_divisor = 4294967291u;

 private:
  uint32_t hash_x_;
  uint32_t hash_y_;
};

int main() {
  using Config = cuda::std::array<std::uint8_t, 20>;
  using V = int;

  const auto sentinel_key = Config{0};
  const auto sentinel_value = 0;

  const std::size_t capacity = 5;

  const std::size_t num_keys = 10;
  thrust::device_vector<Config> keys(num_keys);
  for (std::size_t i = 0; i < num_keys; i++) {
    keys[i] = Config{static_cast<std::uint8_t>(i)};
  }

  bght::bcht<Config, V, array_hash<Config>> table(capacity, sentinel_key, sentinel_value);

  // for simplicity launch one block per key and set the block size to tile/bucket size
  const auto block_size = bght::bcht<Config, V>::bucket_size;
  test_kernel<<<keys.size(), block_size>>>(table, keys.data().get());

  cuda_try(cudaDeviceSynchronize());
}