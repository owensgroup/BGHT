#pragma once

namespace bght {
template <typename First, typename Second>
struct alignas(8) pair {
  // struct pair {
  using first_type = First;
  using second_type = Second;
  First first;
  Second second;
  pair() = default;
  ~pair() = default;
  pair(pair const&) = default;
  pair(pair&&) = default;
  pair& operator=(pair const&) = default;
  pair& operator=(pair&&) = default;

  // bool operator==(const pair& rhs) {
  //  bool is_same = this->first == rhs.first;

  //  //&&this->second == rhs.second;
  //  printf("{%ull, %f} == {%ull, %f}%i\n",
  //         this->first,
  //         this->second,
  //         rhs.first,
  //         rhs.second,
  //         (int)(is_same));
  //  return is_same;
  //}
  // bool operator!=(const pair& rhs) { return !(*this == rhs); }

  __host__ __device__ constexpr pair(First const& f, Second const& s)
      : first{f}, second{s} {}
};

template <class T = void>
struct equal_to {
  constexpr bool operator()(const T& lhs, const T& rhs) const { return lhs == rhs; }
};

}  // namespace bght
