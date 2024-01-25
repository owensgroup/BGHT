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
#include <bght/bcht.hpp>
#include <memory>

template <typename T1, typename T2>
void CHECK(T1 t1, T2 t2) {
  if (t1 != t2) {
    std::cerr << t1 << "!=" << t2 << std::endl;
    std::terminate();
  }
}

template <class T>
struct managed_allocator {
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;

  template <class U>
  struct rebind {
    typedef managed_allocator<U> other;
  };
  managed_allocator() = default;
  template <class U>
  constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* p = nullptr;
    cuda_try(cudaMallocManaged(&p, n * sizeof(T)));
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { cuda_try(cudaFree(p)); }
};

template <typename K, typename V>
void test() {
  using pair_type = bght::pair<K, V>;

  using hasher = bght::universal_hash<K>;
  using equal = bght::equal_to<K>;
  const auto scope = cuda::thread_scope_system;

  // allocators
  using char_allocator_type = managed_allocator<char>;
  using pair_allocator_type =
      typename std::allocator_traits<char_allocator_type>::rebind_alloc<pair_type>;
  using key_allocator_type =
      typename std::allocator_traits<char_allocator_type>::rebind_alloc<K>;
  using value_allocator_type =
      typename std::allocator_traits<char_allocator_type>::rebind_alloc<V>;

  char_allocator_type char_allocator;
  pair_allocator_type pair_allocator{char_allocator};
  key_allocator_type key_allocator{char_allocator};
  value_allocator_type value_allocator{char_allocator};

  thrust::device_vector<pair_type, pair_allocator_type> pairs(4, pair_allocator);
  pairs[0] = pair_type{static_cast<K>(1), static_cast<K>(10)};
  pairs[1] = pair_type{static_cast<K>(2), static_cast<K>(20)};
  pairs[2] = pair_type{static_cast<K>(3), static_cast<K>(30)};
  pairs[3] = pair_type{static_cast<K>(4), static_cast<K>(40)};

  bght::bcht<K, V, hasher, equal, scope, char_allocator_type> table(
      12ull, static_cast<K>(0), static_cast<V>(0), char_allocator);

  bool success = table.insert(pairs.data(), pairs.data() + pairs.size());
  assert(success);

  cuda_try(cudaDeviceSynchronize());

  thrust::device_vector<K, key_allocator_type> queries(6, key_allocator);
  queries[0] = static_cast<K>(1);
  queries[1] = static_cast<K>(3);
  queries[2] = static_cast<K>(5);
  queries[3] = static_cast<K>(2);
  queries[4] = static_cast<K>(4);
  queries[5] = static_cast<K>(12);

  thrust::device_vector<V, value_allocator_type> results(queries.size(), value_allocator);

  table.find(queries.data(), queries.data() + queries.size(), results.begin());

  cuda_try(cudaDeviceSynchronize());

  CHECK(results[0], static_cast<V>(10));
  CHECK(results[1], static_cast<V>(30));
  CHECK(results[2], static_cast<V>(0));
  CHECK(results[3], static_cast<V>(20));
  CHECK(results[4], static_cast<V>(40));
  CHECK(results[5], static_cast<V>(0));
  std::cout << "Success\n";
}

int main() {
  using K = uint32_t;
  using V = uint32_t;
  test<K, V>();
  return 0;
}
