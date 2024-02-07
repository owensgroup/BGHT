# [BGHT: Better GPU Hash Tables](https://owensgroup.github.io/BGHT/)

| [**Documentation**](https://owensgroup.github.io/BGHT/) | [**Examples**](https://github.com/owensgroup/BGHT/tree/main/test)  | [**Examples**](https://github.com/owensgroup/BGHT/tree/main/examples)  |  [**Benchmarks**](https://github.com/owensgroup/BGHT/tree/main/benchmarks) | [**Results**](https://github.com/owensgroup/BGHT/blob/main/results.md) |
|--------------|----------------------|-------------------|-------------------|-------------------|

BGHT is a collection of high-performance static GPU hash tables. BGHT contains hash tables that use three different probing schemes 1) bucketed cuckoo, 2) power-of-two, 3) iceberg hashing. Our bucketed static cuckoo hash table is the state-of-art static hash table.
For more information, please check our papers:

[**Better GPU Hash Tables**](https://owensgroup.github.io/BGHT/) [[arXiv]](https://arxiv.org/abs/2108.07232) [[APOCS]](https://escholarship.org/uc/item/6cb1q6rz)<br>
*[Muhammad A. Awad](https://maawad.github.io/), [Saman Ashkiani](https://scholar.google.com/citations?user=Z4_ZfiEAAAAJ&hl=en), [Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), [Martín Farach-Colton](https://people.cs.rutgers.edu/~farach/), and [John D. Owens](https://www.ece.ucdavis.edu/~jowens/)*

## Key features
* State-of-the-art static GPU hash tables
* Device and host side APIs
* Support for different types of keys and values
* Standard-like APIs

## How to use
BGHT is a header-only library. To use the library, you can add it as a submodule or use [CMake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake) to fetch the library into your CMake-based project ([complete example](https://github.com/owensgroup/BGHT/tree/main/examples/cpm)).
```
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
CPMAddPackage(
  NAME bght
  GITHUB_REPOSITORY owensgroup/BGHT
  GIT_TAG main
  OPTIONS
     "build_tests OFF"
     "build_benchmarks OFF"
)
target_link_libraries(my_library PRIVATE bght)
```

### APIs
All the data structures follow the C++ standard hash map (`std::unordered_map`) APIs closely. An example APIs for BCHT is shown below:
```c++
template <class Key,
          class T,
          class Hash = bght::universal_hash<Key>,
          class KeyEqual = bght::equal_to<Key>,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          class Allocator = bght::cuda_allocator<char>,
          int B = 16> class bcht;
```
#### Member functions
```c++
// Constructor
bcht(std::size_t capacity,
     Key sentinel_key,
     T sentinel_value,
     Allocator const& allocator = Allocator{});
// Host-side APIs
template <typename InputIt>
  bool insert(InputIt first, InputIt last, cudaStream_t stream = 0);
template <typename InputIt, typename OutputIt>
  void find(InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream = 0);
// Device-side APIs
template <typename tile_type>
__device__ bool insert(value_type const& pair, tile_type const& tile);
template <typename tile_type>
__device__ mapped_type find(key_type const& key, tile_type const& tile);
```
### Member types
```
Member type                     Definition
key_type                        Key
mapped_type                     T
value_type                      bght::pair<Key, T>
allocator_type                  Allocator
bucket_size                     Bucket size for device-side APIs cooperative groups tile construction
```


#### Example
```c++
// Example using host-side APIs
#include <bght/cht.hpp>
int main(){
  using key_type = uint32_t;
  using value_type = uint32_t;
  using pair_type = bght::pair<key_type, value_type>;
  std::size_t capacity = 128; std::size_t num_keys = 64;
  key_type invalid_key = 0; value_type invalid_value = 0; // sentinel key and value

  bght::bcht<key_type, value_type> table(capacity, invalid_key, invalid_value); //ctor

  pair_type* pairs; // input pairs
  // ... allocate pairs

  bool success = table.insert(pairs, pairs + num_keys);
  assert(success);

  key_type* queries;  // query keys
  value_type* results; // query result
  // ... allocate queries and results
  table.find(queries, queries + num_keys, results);
}

```
```c++
// Example using device-side APIs
template<class HashMap>
__global__ void kernel(HashMap table){
  // construct tile
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<HashMap::bucket_size>(block);
  pair_type pair{...};
  table.insert(pair, tile);
  pair_type query{..};
  query.second = table.find(query.first, tile);
}
int main(){
  // Call the hash table constructor on the CPU
  bght::bcht<key_type, value_type> table(...);
  // Pass the hash table to a GPU kernel
  kernel<<<...>>>(table);
}
```

## Requirements and limitations
Please create an issue if you face challenges with any of the following limitations and requirements.
### Requirements
* C++17/CUDA C++17
* NVIDIA Volta GPU or later microarchitectures
* CMake 3.8 or later
* CUDA 11.5 or later

#### Using Docker
We provide a docker image that include the software requirements (except for CUDA drivers). To build the docker image, run:
```bash
source docker/build
```
To start the container, run:
```bash
source docker/run
```
After starting the container, you can build and execute BGHT code without any additional requirements.

### limitations
* Currently hash tables based on cuckoo hashing do not support concurrent insertion and queries. IHT and P2BHT support concurrent insertions and queries.
For hash tables that use a probing scheme other than IHT:
* Keys must be unique.
* Construction of the data structures offered *may* fail. In these scenarios, reconstructing the table using a larger capacity or a lower load factor should be considered. Our paper offers recommended hash table load factors (for uniformly distributed unsigned keys) to achieve at least a 99% success rate ([See Fig. 2](https://arxiv.org/abs/2108.07232)). For example, BCHT will offer a 100% success rate for up to 0.991 load factor. Please create an issue if you encounter any problems with different key distributions.

## Reproducing the arXiv paper results
To reproduce the results, follow the following [steps](reproduce.md). You can also view our results [here](./results.md). If you find any mismatch (either faster or slower) between the results offered in the repository or the paper, please create an issue, and we will investigate the performance changes.

## Benchmarks
Please check our [paper](https://arxiv.org/abs/2108.07232) for comprehensive analysis and benchmarks. Also, see the following steps to [reproduce](reproduce.md) the results.

## Questions and bug report
Please create an issue. We will welcome any contributions that improve the usability and quality of our repository.

## Bibtex
```bibtex
@InProceedings{   Awad:2023:AAI,
  title         = {Analyzing and Implementing {GPU} Hash Tables},
  author        = {Muhammad A. Awad and Saman Ashkiani and Serban D.
                  Porumbescu and Mart{\'{i}}n Farach-Colton and John D.
                  Owens},
  booktitle     = "SIAM Symposium on Algorithmic Principles of Computer
                  Systems",
  series        = {APOCS23},
  year          = 2023,
  month         = jan,
  pages         = {33--50},
  code          = {https://github.com/owensgroup/BGHT},
  doi           = {10.1137/1.9781611977578.ch3},
  url           = {https://escholarship.org/uc/item/6cb1q6rz}
}
```

## Acknowledgments

The structure and organization of the repository were inspired by [NVIDIA's cuCollection](https://github.com/nviDIA/cuCollections/) and [RXMesh](https://github.com/owensgroup/RXMesh).
