#pragma once
namespace bght {
namespace detail {
struct mars_rng_32 {
  uint32_t y;
  __host__ __device__ constexpr mars_rng_32() : y(2463534242) {}
  constexpr uint32_t __host__ __device__ operator()() {
    y ^= (y << 13);
    y = (y >> 17);
    return (y ^= (y << 5));
  }
};
}  // namespace detail
}  // namespace bght
