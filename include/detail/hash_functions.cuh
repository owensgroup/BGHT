#pragma once
namespace bght {
namespace detail {
template <typename Key>
struct universal_hash {
  using key_type = Key;
  using result_type = Key;
  __host__ __device__ constexpr universal_hash(uint32_t hash_x, uint32_t hash_y)
      : hash_x_(hash_x), hash_y_(hash_y) {}

  constexpr result_type __host__ __device__ operator()(const key_type& key) const {
    return (((hash_x_ ^ key) + hash_y_) % prime_divisor);
  }

  universal_hash(const universal_hash&) = default;
  universal_hash() = default;
  universal_hash(universal_hash&&) = default;
  universal_hash& operator=(universal_hash const&) = default;
  universal_hash& operator=(universal_hash&&) = default;
  ~universal_hash() = default;

  static constexpr uint32_t prime_divisor = 4294967291u;

 private:
  uint32_t hash_x_;
  uint32_t hash_y_;
};

template <typename Hash, typename RNG>
Hash initialize_hf(RNG& rng) {
  uint32_t x = rng() % Hash::prime_divisor;
  if (x < 1u) {
    x = 1;
  }
  uint32_t y = rng() % Hash::prime_divisor;
  return Hash(x, y);
}
}  // namespace detail
}  // namespace bght
