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
#include <cuda_runtime.h>
#include <bght/detail/cuda_helpers.cuh>
namespace bght {
template <typename T>
struct cuda_deleter {
  void operator()(T* p) { cuda_try(cudaFree(p)); }
};

template <class T>
struct cuda_allocator {
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;

  template <class U>
  struct rebind {
    typedef cuda_allocator<U> other;
  };
  cuda_allocator() = default;
  template <class U>
  constexpr cuda_allocator(const cuda_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* p = nullptr;
    cuda_try(cudaMalloc(&p, n * sizeof(T)));
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t n) noexcept { cuda_try(cudaFree(p)); }
};
}  // namespace bght
