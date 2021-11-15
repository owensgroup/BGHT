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
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <limits>
#include <rkg.hpp>
#include <type_traits>

#include <cht.hpp>

#include <bcht.hpp>
#include <iht.hpp>
#include <p2bht.hpp>

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

template <class K, class V>
using iht16_80 = iht16<K, V, 12>;

template <class K, class V>
using iht32_80 = iht32<K, V, 25>;

int main() {
  std::cout << "Testing bcht8\n";
  test_scheme<bcht8>();
  std::cout << "Testing bcht16\n";
  test_scheme<bcht16>();
  std::cout << "Testing bcht32\n";
  test_scheme<bcht32>();

  std::cout << "Testing iht16_80\n";
  test_scheme<iht16_80>();
  std::cout << "Testing iht32_80\n";
  test_scheme<iht32_80>();

  std::cout << "Testing p2bht16\n";
  test_scheme<p2bht16>();
  std::cout << "Testing p2bht32\n";
  test_scheme<p2bht32>();

  std::cout << "Success\n";

  return 0;
}
