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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <limits>
#include <vector>
#include "bght/bcht.hpp"

template <typename T1, typename T2>
void CHECK(T1 t1, T2 t2) {
  if (t1 != t2) {
    std::cerr << t1 << "!=" << t2 << std::endl;
    std::terminate();
  }
}

int main() {
  using K = unsigned;
  using V = int;
  using pair_type = bght::pair<K, V>;

  auto pair0 = pair_type{static_cast<K>(1), static_cast<V>(5)};
  auto pair1 = pair_type{static_cast<K>(2), static_cast<V>(6)};
  auto pair2 = pair_type{static_cast<K>(3), static_cast<V>(7)};
  auto pair3 = pair_type{static_cast<K>(4), static_cast<V>(8)};

  std::vector<pair_type> h_pairs{pair0, pair1, pair2, pair3};
  thrust::device_vector<pair_type> d_pairs(h_pairs);

  auto sentinel_key = std::numeric_limits<K>::max();
  auto sentinel_value = std::numeric_limits<V>::max();
  bght::bcht<K, V> table(64ull, sentinel_key, sentinel_value);

  bool success =
      table.insert(d_pairs.data().get(), d_pairs.data().get() + d_pairs.size());
  if (!success) {
    std::cout << "build failed\n" << std::endl;
    std::terminate();
  }

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
  std::cout << "Success\n";
}
