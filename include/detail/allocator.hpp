#pragma once
#include <cuda_runtime.h>
#include <detail/cuda_helpers.cuh>
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
