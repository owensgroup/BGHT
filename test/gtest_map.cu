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

typedef testing::Types<MakeHashMapData<bght::bcht, uint32_t, uint32_t>,
                       MakeHashMapData<bght::bcht, uint32_t, uint64_t>,
                       MakeHashMapData<bght::bcht, uint32_t, char>,
                       MakeHashMapData<bght::bcht, uint32_t, uint32_t*>,
                       MakeHashMapData<bght::bcht, uint64_t, uint32_t>,
                       MakeHashMapData<bght::bcht, uint64_t, uint64_t>,
                       MakeHashMapData<bght::bcht, uint64_t, char>,
                       MakeHashMapData<bght::bcht, uint64_t, uint32_t*>,

                       MakeHashMapData<bght::iht, uint32_t, uint32_t>,
                       MakeHashMapData<bght::iht, uint32_t, uint64_t>,
                       MakeHashMapData<bght::iht, uint32_t, char>,
                       MakeHashMapData<bght::iht, uint32_t, uint32_t*>,
                       MakeHashMapData<bght::iht, uint64_t, uint32_t>,
                       MakeHashMapData<bght::iht, uint64_t, uint64_t>,
                       MakeHashMapData<bght::iht, uint64_t, char>,
                       MakeHashMapData<bght::iht, uint64_t, uint32_t*>,

                       MakeHashMapData<bght::p2bht, uint32_t, uint32_t>,
                       MakeHashMapData<bght::p2bht, uint32_t, uint64_t>,
                       MakeHashMapData<bght::p2bht, uint32_t, char>,
                       MakeHashMapData<bght::p2bht, uint32_t, uint32_t*>,
                       MakeHashMapData<bght::p2bht, uint64_t, uint32_t>,
                       MakeHashMapData<bght::p2bht, uint64_t, uint64_t>,
                       MakeHashMapData<bght::p2bht, uint64_t, char>,
                       MakeHashMapData<bght::p2bht, uint64_t, uint32_t*>>
    Implementations;

TYPED_TEST_SUITE(HashMapTest, Implementations);

TYPED_TEST(HashMapTest, Construction) {
  auto last_error = cudaPeekAtLastError();
  EXPECT_TRUE(last_error == cudaSuccess);
}

TYPED_TEST(HashMapTest, InsertSuccess) {
  std::size_t num_keys = 4;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;
  testing_input<key_type, value_type, pair_type> input(num_keys);
  bool insert_result =
      this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);
  EXPECT_EQ(insert_result, true);
  input.free();
}

TYPED_TEST(HashMapTest, EmptySize) {
  auto size = this->hashmap_->size();
  EXPECT_TRUE(size == 0ull);
}

TYPED_TEST(HashMapTest, Clear) {
  this->hashmap_->clear();
  auto last_error = cudaPeekAtLastError();
  EXPECT_TRUE(last_error == cudaSuccess);
}

TYPED_TEST(HashMapTest, InsertSize) {
  std::size_t num_keys = 4;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;

  testing_input<key_type, value_type, pair_type> input(num_keys);
  this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);
  auto size = this->hashmap_->size();
  EXPECT_TRUE(size == num_keys);
  input.free();
}

TYPED_TEST(HashMapTest, SizeAfterClear) {
  std::size_t num_keys = 4;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;

  testing_input<key_type, value_type, pair_type> input(num_keys);
  this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);

  this->hashmap_->clear();
  auto size = this->hashmap_->size();
  EXPECT_TRUE(size == 0ull);
  input.free();
}

TYPED_TEST(HashMapTest, FindExist) {
  std::size_t num_keys = 4;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;
  mapped_vector<value_type> find_results(num_keys);
  testing_input<key_type, value_type, pair_type> input(num_keys);
  this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);
  this->hashmap_->find(
      input.keys_exist.data(), input.keys_exist.data() + num_keys, find_results.data());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.pairs[i].second;
    auto found_value = find_results[i];
    EXPECT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

TYPED_TEST(HashMapTest, FindNotExist) {
  std::size_t num_keys = 4;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;
  mapped_vector<value_type> find_results(num_keys);
  testing_input<key_type, value_type, pair_type> input(num_keys);
  this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);
  this->hashmap_->find(input.keys_not_exist.data(),
                       input.keys_not_exist.data() + num_keys,
                       find_results.data());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = TestFixture::map_data::sentinel_value;
    auto found_value = find_results[i];
    EXPECT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

TYPED_TEST(HashMapTest, InsertSentinel) {
  std::size_t num_keys = 1;
  using key_type = typename TestFixture::map_data::hash_map::key_type;
  using value_type = typename TestFixture::map_data::hash_map::mapped_type;
  using pair_type = typename TestFixture::map_data::hash_map::value_type;
  bool contain_sentinel = true;
  testing_input<key_type, value_type, pair_type> input(num_keys, contain_sentinel);
  bool success =
      this->hashmap_->insert(input.pairs.data(), input.pairs.data() + num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(success, false);
  input.free();
}
// other tests to add:
// custom types
// custom allocator
// large number of keys and high load factor

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}