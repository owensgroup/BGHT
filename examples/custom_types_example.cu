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

#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <bght/cmd.hpp>
#include <bght/gpu_timer.hpp>
#include <bght/rkg.hpp>
#include <limits>
#include <type_traits>

#include <bght/cht.hpp>

#include <bght/bcht.hpp>
#include <bght/iht.hpp>
#include <bght/p2bht.hpp>

template <class K, class V>
using iht16_80 = bght::iht16<K, V, 12>;

template <class K, class V>
using iht32_80 = bght::iht32<K, V, 25>;

template <typename T1, typename T2>
void CHECK(T1 t1, T2 t2) {
  assert(t1 == t2);
}
template <typename T1, typename T2, template <class...> class HashMap>
struct unit_test {
  static void test() {
    using K = T1;
    using V = T2;
    using pair_type = bght::pair<K, V>;
    auto pair0 = pair_type{static_cast<K>(1), static_cast<V>(5)};
    auto pair1 = pair_type{static_cast<K>(2), static_cast<V>(6)};
    auto pair2 = pair_type{static_cast<K>(3), static_cast<V>(7)};
    auto pair3 = pair_type{static_cast<K>(4), static_cast<V>(8)};
    std::vector<pair_type> h_pairs{pair0, pair1, pair2, pair3};
    thrust::device_vector<pair_type> d_pairs(h_pairs);

    auto sentinel_key = std::numeric_limits<K>::max();
    auto sentinel_value = std::numeric_limits<V>::max();
    HashMap<K, V> table(12ull, sentinel_key, sentinel_value);

    bool success =
        table.insert(d_pairs.data().get(), d_pairs.data().get() + d_pairs.size());
    assert(success);

    thrust::device_vector<K> d_queries(std::vector<K>{static_cast<K>(1),
                                                      static_cast<K>(3),
                                                      static_cast<K>(5),
                                                      static_cast<K>(2),
                                                      static_cast<K>(4),
                                                      static_cast<K>(12)});
    thrust::device_vector<V> d_results(d_queries.size());

    table.find(d_queries.data().get(),
               d_queries.data().get() + d_queries.size(),
               d_results.begin());

    thrust::host_vector<V> h_results = d_results;
    CHECK(h_results[0], static_cast<V>(5));
    CHECK(h_results[1], static_cast<V>(7));
    CHECK(h_results[2], static_cast<V>(sentinel_value));
    CHECK(h_results[3], static_cast<V>(6));
    CHECK(h_results[4], static_cast<V>(8));
    CHECK(h_results[5], static_cast<V>(sentinel_value));
  }
};

template <typename T, template <class...> class HashMap>
struct unit_test<T, char, HashMap> {
  static void test() {
    using K = T;
    using V = char;
    using pair_type = bght::pair<K, V>;
    auto pair0 = pair_type{static_cast<K>(1), static_cast<V>('a')};
    auto pair1 = pair_type{static_cast<K>(2), static_cast<V>('b')};
    auto pair2 = pair_type{static_cast<K>(3), static_cast<V>('c')};
    auto pair3 = pair_type{static_cast<K>(4), static_cast<V>('d')};
    std::vector<pair_type> h_pairs{pair0, pair1, pair2, pair3};
    thrust::device_vector<pair_type> d_pairs(h_pairs);

    HashMap<K, V> table(12ull, static_cast<K>(0), '\0');

    bool success =
        table.insert(d_pairs.data().get(), d_pairs.data().get() + d_pairs.size());
    assert(success);

    thrust::device_vector<K> d_queries(std::vector<K>{static_cast<K>(1),
                                                      static_cast<K>(3),
                                                      static_cast<K>(5),
                                                      static_cast<K>(2),
                                                      static_cast<K>(4),
                                                      static_cast<K>(12)});
    thrust::device_vector<V> d_results(d_queries.size());

    table.find(d_queries.data().get(),
               d_queries.data().get() + d_queries.size(),
               d_results.begin());

    thrust::host_vector<V> h_results = d_results;
    CHECK(h_results[0], static_cast<V>('a'));
    CHECK(h_results[1], static_cast<V>('c'));
    CHECK(h_results[2], static_cast<V>('\0'));
    CHECK(h_results[3], static_cast<V>('b'));
    CHECK(h_results[4], static_cast<V>('d'));
    CHECK(h_results[5], static_cast<V>('\0'));
  }
};

template <typename T, template <class...> class HashMap>
struct unit_test<T, T*, HashMap> {
  static void test() {
    using K = T;
    using V = T*;

    thrust::device_vector<K> d_queries(std::vector<K>{static_cast<K>(1),
                                                      static_cast<K>(3),
                                                      static_cast<K>(5),
                                                      static_cast<K>(2),
                                                      static_cast<K>(4),
                                                      static_cast<K>(12)});
    thrust::device_vector<V> d_results(d_queries.size());

    using pair_type = bght::pair<K, V>;
    auto pair0 = pair_type{static_cast<K>(1), d_queries.data().get() + 1};
    auto pair1 = pair_type{static_cast<K>(2), d_queries.data().get() + 2};
    auto pair2 = pair_type{static_cast<K>(3), d_queries.data().get() + 3};
    auto pair3 = pair_type{static_cast<K>(4), d_queries.data().get() + 4};

    std::vector<pair_type> h_pairs{pair0, pair1, pair2, pair3};
    thrust::device_vector<pair_type> d_pairs(h_pairs);

    bght::bcht<K, V> table(12ull, 0u, nullptr);

    bool success =
        table.insert(d_pairs.data().get(), d_pairs.data().get() + d_pairs.size());
    assert(success);
    table.find(d_queries.data().get(),
               d_queries.data().get() + d_queries.size(),
               d_results.begin());

    thrust::host_vector<V> h_results = d_results;
    CHECK(h_results[0], d_queries.data().get() + 1);
    CHECK(h_results[1], d_queries.data().get() + 3);
    CHECK(h_results[2], nullptr);
    CHECK(h_results[3], d_queries.data().get() + 2);
    CHECK(h_results[4], d_queries.data().get() + 4);
    CHECK(h_results[5], nullptr);
  }
};

__device__ inline unsigned get_sm_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

// Testing passing a hashmap to the device
template <typename HashMap>
__global__ void test_kernel(HashMap map) {
  using pair_type = typename HashMap::value_type;
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  // tile
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);

  // pair to insert
  auto tile_id = thread_id / HashMap::bucket_size;
  auto sm_id = get_sm_id();
  pair_type pair{tile_id, sm_id};

  // insert
  map.insert(pair, tile);

  // Note that we currently don't support concurrent insertions and queries
  // however, this test should succeed

  // lookup
  auto find_result = map.find(pair.first, tile);

  // check result
  assert(find_result == sm_id);
}

template <typename K, typename V, template <class...> class HashMap>
void pass_to_kernel_test() {
  auto sentinel_key = std::numeric_limits<K>::max();
  auto sentinel_value = std::numeric_limits<V>::max();
  HashMap<K, V> table(12ull, sentinel_key, sentinel_value);

  test_kernel<<<1, 32>>>(table);
}

// Testing different built-in types
template <template <class...> class HashMap>
void test_scheme() {
  unit_test<uint32_t, char, HashMap>::test();
  unit_test<uint64_t, char, HashMap>::test();

  unit_test<uint32_t, float, HashMap>::test();
  unit_test<uint32_t, double, HashMap>::test();

  unit_test<uint64_t, float, HashMap>::test();
  unit_test<uint64_t, double, HashMap>::test();

  unit_test<uint32_t, uint32_t*, HashMap>::test();
  unit_test<uint64_t, uint64_t*, HashMap>::test();

  pass_to_kernel_test<uint32_t, unsigned, HashMap>();
}

// Testing custom keys
struct custom_key {
  std::size_t x;
  uint32_t y;
  uint64_t z;

  custom_key(const custom_key&) = default;
  custom_key() = default;
  custom_key(custom_key&&) = default;
  custom_key& operator=(custom_key const&) = default;
  custom_key& operator=(custom_key&&) = default;

  constexpr bool operator==(const custom_key& rhs) const {
    return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z);
  }
  constexpr bool operator!=(const custom_key& rhs) const { return !(*this == rhs); }
};

struct custom_equal {
  constexpr bool operator()(const custom_key& lhs, const custom_key& rhs) const {
    return lhs == rhs;
  }
};

struct custom_key_hash {
  using key_type = custom_key;
  using result_type = std::size_t;
  constexpr custom_key_hash(uint32_t hash_x, uint32_t hash_y)
      : hash_x_(hash_x), hash_y_(hash_y) {}

  constexpr result_type __host__ __device__ operator()(const key_type& key) const {
    return (((hash_x_ ^ key.x) + hash_y_) % prime_divisor);
  }

  custom_key_hash(const custom_key_hash&) = default;
  custom_key_hash() = default;
  custom_key_hash(custom_key_hash&&) = default;
  custom_key_hash& operator=(custom_key_hash const&) = default;
  custom_key_hash& operator=(custom_key_hash&&) = default;
  ~custom_key_hash() = default;
  static constexpr uint32_t prime_divisor = 4294967291u;

 private:
  uint32_t hash_x_;
  uint32_t hash_y_;
};

void test_custom_type() {
  custom_key invalid_key{0, 0, 0};
  using hasher = custom_key_hash;
  using key_eq = custom_equal;
  bght::bcht<custom_key, int, hasher, custom_equal> table(128ull, invalid_key, 0);
  using pair_type = bght::pair<custom_key, int>;

  std::vector<custom_key> h_keys{{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12},
                                 {13, 14, 15},
                                 {16, 17, 18},
                                 {19, 20, 21}};

  std::vector<pair_type> h_pairs{{h_keys[0], 10},
                                 {h_keys[1], 11},
                                 {h_keys[2], 12},
                                 {h_keys[3], 13},
                                 {h_keys[4], 14},
                                 {h_keys[5], 15},
                                 {h_keys[6], 16}};

  thrust::device_vector<pair_type> d_pairs(h_pairs);

  bool success =
      table.insert(d_pairs.data().get(), d_pairs.data().get() + d_pairs.size());

  assert(success);

  std::vector<custom_key> h_queries{h_keys[0],
                                    h_keys[1],
                                    h_keys[2],
                                    h_keys[3],
                                    {22, 23, 24},
                                    h_keys[4],
                                    {26, 28, 24},
                                    h_keys[5],
                                    h_keys[6]};

  thrust::device_vector<custom_key> d_queries(h_queries);

  thrust::device_vector<int> d_results(d_queries.size(), 0);

  table.find(d_queries.data().get(),
             d_queries.data().get() + d_queries.size(),
             d_results.data().get());

  thrust::host_vector<int> h_results(d_results);

  CHECK(h_results[0], 10);
  CHECK(h_results[1], 11);
  CHECK(h_results[2], 12);
  CHECK(h_results[3], 13);
  CHECK(h_results[4], 0);
  CHECK(h_results[5], 14);
  CHECK(h_results[6], 0);
  CHECK(h_results[7], 15);
  CHECK(h_results[8], 16);
}

template <typename T>
struct P3 {
  T i;
  T j;
  T k;
  using type = T;
  P3(const P3&) = default;
  P3() = default;
  P3(P3&&) = default;
  P3& operator=(P3 const&) = default;
  P3& operator=(P3&&) = default;

  constexpr bool operator==(const P3& rhs) const {
    return (this->i == rhs.i) && (this->j == rhs.j) && (this->k == rhs.k);
  }
  constexpr bool operator!=(const P3& rhs) const { return !(*this == rhs); }
};

using cell_type = P3<uint32_t>;
using color_type = P3<float>;

struct cell_hasher {
  using key_type = cell_type;
  using result_type = std::size_t;
  constexpr cell_hasher(uint32_t hash_x, uint32_t hash_y)
      : hash_x_(hash_x), hash_y_(hash_y) {}

  constexpr result_type __host__ __device__ operator()(const key_type& key) const {
    return hash_one(key.i) | hash_one(key.j) | hash_one(key.k);
  }

  constexpr result_type __host__ __device__ hash_one(const key_type::type& key) const {
    return (((hash_x_ ^ key) + hash_y_) % prime_divisor);
  }

  cell_hasher(const cell_hasher&) = default;
  cell_hasher() = default;
  cell_hasher(cell_hasher&&) = default;
  cell_hasher& operator=(cell_hasher const&) = default;
  cell_hasher& operator=(cell_hasher&&) = default;
  ~cell_hasher() = default;
  static constexpr uint32_t prime_divisor = 4294967291u;

 private:
  uint32_t hash_x_;
  uint32_t hash_y_;
};

struct cell_equal {
  constexpr bool operator()(const cell_type& lhs, const cell_type& rhs) const {
    return lhs == rhs;
  }
};

void test_vec3_like() {
  auto to_value = [] __host__ __device__(cell_type x) {
    return color_type{x.i / 255.0f, x.j / 255.0f, x.k / 255.0f};
  };

  auto invalid_cell = cell_type{0, 0, 0};
  auto invalid_color = color_type{0, 0, 0};
  using pair_type = bght::pair<cell_type, color_type>;

  bght::bcht<cell_type, color_type, cell_hasher, cell_equal> table(
      128ull, invalid_cell, invalid_color);

  std::vector<cell_type> h_cells{{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9},
                                 {10, 11, 12},
                                 {13, 14, 15},
                                 {16, 17, 18},
                                 {19, 20, 21}};

  std::vector<pair_type> h_pairs{{h_cells[0], to_value(h_cells[0])},
                                 {h_cells[1], to_value(h_cells[1])},
                                 {h_cells[2], to_value(h_cells[2])},
                                 {h_cells[3], to_value(h_cells[3])},
                                 {h_cells[4], to_value(h_cells[4])},
                                 {h_cells[5], to_value(h_cells[5])},
                                 {h_cells[6], to_value(h_cells[6])}};

  thrust::device_vector<pair_type> d_pairs(h_pairs);
  thrust::device_vector<cell_type> d_queries(h_cells);
  thrust::device_vector<color_type> d_results(d_queries.size(), invalid_color);

  bool success =
      table.insert(d_pairs.data().get(), d_pairs.data().get() + d_pairs.size());

  assert(success);

  table.find(d_queries.data().get(),
             d_queries.data().get() + d_queries.size(),
             d_results.data().get());
  thrust::host_vector<color_type> h_results(d_results);

  for (std::size_t i = 0; i < d_queries.size(); i++) {
    CHECK(h_results[i], to_value(h_cells[i]));
  }
}

int main() {
  std::cout << "Testing custom type\n";
  test_custom_type();
  std::cout << "Testing vec3-like type\n";
  test_vec3_like();
  std::cout << "Testing bght::bcht8\n";
  test_scheme<bght::bcht8>();
  std::cout << "Testing bght::bcht16\n";
  test_scheme<bght::bcht16>();
  std::cout << "Testing bght::bcht32\n";
  test_scheme<bght::bcht32>();

  std::cout << "Testing bght::iht16_80\n";
  test_scheme<iht16_80>();
  std::cout << "Testing bght::iht32_80\n";
  test_scheme<iht32_80>();

  std::cout << "Testing bght::p2bht16\n";
  test_scheme<bght::p2bht16>();
  std::cout << "Testing bght::p2bht32\n";
  test_scheme<bght::p2bht32>();

  std::cout << "Success\n";

  return 0;
}
