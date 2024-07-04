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
#include <hip/hip_runtime.h>
#include <bght/detail/hip_helpers.hpp>
namespace bght {
template <typename T>
struct hip_deleter {
  void operator()(T* p) { hip_try(hipFree(p)); }
};

template <class T>
struct hip_allocator {
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;

  template <class U>
  struct rebind {
    typedef hip_allocator<U> other;
  };
  hip_allocator() = default;
  template <class U>
  constexpr hip_allocator(const hip_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* p = nullptr;
    hip_try(hipMalloc(&p, n * sizeof(T)));
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { hip_try(hipFree(p)); }
};
}  // namespace bght
