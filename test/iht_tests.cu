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

#include <gtest/gtest.h>

#include <bght/cht.hpp>

#include <bght/bcht.hpp>
#include <bght/iht.hpp>
#include <bght/p2bht.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>

// Based on sample6_unittest
// https://github.com/google/googletest/blob/main/googletest/samples/sample6_unittest.cc

namespace {

template <typename T>
constexpr auto get_sentinel() {
  if constexpr (std::is_pointer<T>::value) {
    return nullptr;
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <class HashMap, std::size_t Capacity, auto SentinelKey, auto SentinelValue>
struct HashMapData {
  using hash_map = HashMap;
  static constexpr auto capacity = Capacity;
  static constexpr auto sentinel_key = SentinelKey;
  static constexpr auto sentinel_value = SentinelValue;
};
template <class MapData>
class HashMapTest : public testing::Test {
 protected:
  HashMapTest() {
    hashmap_ = new typename map_data::hash_map(
        MapData::capacity, MapData::sentinel_key, MapData::sentinel_value);
  }
  ~HashMapTest() override { delete hashmap_; }
  using map_data = MapData;
  typename map_data::hash_map* hashmap_;
};

template <typename T>
struct mapped_vector {
  mapped_vector(std::size_t capacity) : capacity_(capacity) { allocate(capacity); }
  T& operator[](std::size_t index) { return dh_buffer_[index]; }
  ~mapped_vector() {}
  void free() { cuda_try(cudaFreeHost(dh_buffer_)); }
  T* data() const { return dh_buffer_; }

 private:
  void allocate(std::size_t count) {
    cuda_try(cudaMallocHost(&dh_buffer_, sizeof(T) * count));
  }
  std::size_t capacity_;
  T* dh_buffer_;
};

template <class key_type, class value_type, class pair_type>
struct testing_input {
  testing_input(std::size_t input_num_keys, bool contain_sentinel = false)
      : num_keys(input_num_keys)
      , pairs(input_num_keys)
      , keys_exist(input_num_keys)
      , keys_not_exist(input_num_keys) {
    make_input(contain_sentinel);
  }
  void make_input(bool contain_sentinel) {
    for (std::size_t i = 0; i < num_keys; i++) {
      value_type value{};
      if constexpr (std::is_pointer<value_type>::value) {
        value = reinterpret_cast<value_type>(pairs.data() + i);
      } else {
        value = static_cast<value_type>(i);
      }
      key_type key = i;
      if (contain_sentinel) {
        key = get_sentinel<key_type>();
      }
      pairs[i] = {key, value};
      keys_exist[i] = pairs[i].first;
      keys_not_exist[i] = pairs[i].first + static_cast<key_type>(num_keys);
    }
  }
  void free() {
    pairs.free();
    keys_exist.free();
    keys_not_exist.free();
  }

  std::size_t num_keys;
  mapped_vector<pair_type> pairs;
  mapped_vector<key_type> keys_exist;
  mapped_vector<key_type> keys_not_exist;
};

template <template <class...> class HashMap, typename K, class V>
using MakeHashMapData =
    HashMapData<HashMap<K, V>, 512ull, get_sentinel<K>(), get_sentinel<V>()>;

typedef testing::Types<MakeHashMapData<bght::iht, uint32_t, uint32_t>,
                       MakeHashMapData<bght::iht, uint32_t, uint64_t>,
                       MakeHashMapData<bght::iht, uint32_t, char>,
                       MakeHashMapData<bght::iht, uint64_t, uint32_t>,
                       MakeHashMapData<bght::iht, uint64_t, uint64_t>,
                       MakeHashMapData<bght::iht, uint64_t, char>>
    Implementations;

TYPED_TEST_SUITE(HashMapTest, Implementations);

TYPED_TEST(HashMapTest, Construction) {
  auto last_error = cudaPeekAtLastError();
  EXPECT_TRUE(last_error == cudaSuccess);
}

TYPED_TEST(HashMapTest, Empty) {
  auto empty = this->hashmap_->empty();
  EXPECT_TRUE(empty == true);
}

TYPED_TEST(HashMapTest, Clear) {
  this->hashmap_->clear();
  auto last_error = cudaPeekAtLastError();
  EXPECT_TRUE(last_error == cudaSuccess);

  auto empty = this->hashmap_->empty();
  EXPECT_TRUE(empty == true);
}

TYPED_TEST(HashMapTest, NotEmpty) {
  std::size_t num_keys = 4;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;

  auto empty = this->hashmap_->empty();
  EXPECT_TRUE(empty == true);

  testing_input<key_type, value_type, pair_type> input(num_keys);
  this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);
  auto size = this->hashmap_->size();
  EXPECT_TRUE(size == num_keys);

  empty = this->hashmap_->empty();
  EXPECT_TRUE(empty == false);

  input.free();
}

template <typename HashMap>
__global__ void contains_kernel(HashMap map) {
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  using key_type = typename HashMap::key_type;
  using value_type = typename HashMap::mapped_type;
  using pair_type = typename HashMap::value_type;

  pair_type pair0(key_type{1}, value_type{3});
  pair_type pair1(key_type{5}, value_type{24});

  auto insert_success = map.insert(pair0, tile).second;
  assert(insert_success);

  auto key0_exist = map.contains(pair0.first, tile);
  assert(key0_exist);

  auto key1_exist = map.contains(pair1.first, tile);
  assert(!key1_exist);
}

TYPED_TEST(HashMapTest, Contains) {
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;

  const auto bucket_size = TestFixture::map_data::hash_map::bucket_size;
  contains_kernel<<<1, bucket_size>>>(*this->hashmap_);

  auto sync_success = cudaDeviceSynchronize();

  EXPECT_EQ(sync_success, cudaSuccess);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}