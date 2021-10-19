#pragma once
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
namespace rkg {
template <typename key_type, typename value_type>
value_type obvious_bit_reversal(key_type in) {
  // https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
  unsigned int v = (unsigned int)in;  // input bits to be reversed
  unsigned int r = v;                 // r will be reversed bits of v; first get LSB of v
  int s = sizeof(v) * CHAR_BIT - 1;   // extra shift needed at end

  for (v >>= 1; v; v >>= 1) {
    r <<= 1;
    r |= v & 1;
    s--;
  }
  r <<= s;  // shift when v's highest bits are zero
  return value_type(r);
}

template <typename key_type, typename value_type>
value_type generate_value(key_type in) {
  return in + 1;
  // return obvious_bit_reversal<key_type, value_type>(in);
}

template <typename key_type, typename value_type, typename size_type>
void generate_uniform_unique_pairs(std::vector<key_type>& keys,
                                   std::vector<value_type>& values,
                                   size_type num_keys,
                                   bool cache = false,
                                   key_type min_key = 0) {
  keys.resize(num_keys);
  values.resize(num_keys);
  unsigned seed = 1;
  // bool cache = true;
  std::string dataset_dir = "dataset";
  std::string dataset_name = std::to_string(num_keys) + "_" + std::to_string(seed);
  std::string dataset_path = dataset_dir + "/" + dataset_name;
  if (cache) {
    if (std::filesystem::exists(dataset_dir)) {
      if (std::filesystem::exists(dataset_path)) {
        std::cout << "Reading cached keys.." << std::endl;
        std::ifstream dataset(dataset_path, std::ios::binary);
        dataset.read((char*)keys.data(), sizeof(key_type) * num_keys);
        dataset.read((char*)values.data(), sizeof(value_type) * num_keys);
        dataset.close();
        return;
      }
    } else {
      std::filesystem::create_directory(dataset_dir);
    }
  }
  std::random_device rd;
  std::mt19937 rng(seed);
  auto max_key = std::numeric_limits<key_type>::max() - 1;
  std::uniform_int_distribution<key_type> uni(min_key, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) {
    unique_keys.insert(uni(rng));
    // unique_keys.insert(unique_keys.size() + 1);
  }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());
  std::shuffle(keys.begin(), keys.end(), rng);
  // std::transform(
  //    keys.cbegin(), keys.cend(), values.begin(), generate_value<key_type, value_type>);

#ifdef _WIN32
  // OpenMP + windows don't allow unsigned loops
  for (uint32_t i = 0; i < unique_keys.size(); i++) {
    values[i] = generate_value<key_type, value_type>(keys[i]);
  }
#else
#pragma omp parallel for
  for (uint32_t i = 0; i < unique_keys.size(); i++) {
    values[i] = generate_value<key_type, value_type>(keys[i]);
  }
#endif

  if (cache) {
    std::cout << "Caching.." << std::endl;
    std::ofstream dataset(dataset_path, std::ios::binary);
    dataset.write((char*)keys.data(), sizeof(key_type) * num_keys);
    dataset.write((char*)values.data(), sizeof(value_type) * num_keys);
    dataset.close();
  }
}

template <typename key_type, typename size_type>
void generate_uniform_unique_keys(std::vector<key_type>& keys, size_type num_keys) {
  keys.resize(num_keys);
  unsigned seed = 1;
  std::random_device rd;
  std::mt19937 rng(seed);
  auto max_key = std::numeric_limits<key_type>::max() - 1;
  std::uniform_int_distribution<key_type> uni(0, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) {
    unique_keys.insert(uni(rng));
  }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());
  std::shuffle(keys.begin(), keys.end(), rng);
}

}  // namespace rkg
