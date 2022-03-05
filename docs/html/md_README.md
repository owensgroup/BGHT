::: {#top}
::: {#titlearea}
+-----------------------------------------------------------------------+
| ::: {#projectname}                                                    |
| BGHT                                                                  |
| :::                                                                   |
|                                                                       |
| ::: {#projectbrief}                                                   |
| Better GPU Hash Tables                                                |
| :::                                                                   |
+-----------------------------------------------------------------------+
:::

::: {#main-nav}
:::

::: {#MSearchSelectWindow onmouseover="return searchBox.OnSearchSelectShow()" onmouseout="return searchBox.OnSearchSelectHide()" onkeydown="return searchBox.OnSearchSelectKey(event)"}
:::

::: {#MSearchResultsWindow}
:::
:::

<div>

::: header
::: headertitle
::: title
BGHT: Better GPU Hash Tables
:::
:::
:::

::: contents
::: textblock
  **[Examples/Tests](https://github.com/owensgroup/BGHT/tree/main/test)**   **[Benchmarks](https://github.com/owensgroup/BGHT/tree/main/benchmarks)**   **[Results](https://github.com/owensgroup/BGHT/blob/main/results.md)**
  ------------------------------------------------------------------------- --------------------------------------------------------------------------- ------------------------------------------------------------------------

BGHT is a collection of high-performance static GPU hash tables. BGHT
contains hash tables that use three different probing schemes 1)
bucketed cuckoo, 2) power-of-two, 3) iceberg hashing. Our bucketed
static cuckoo hash table is the state-of-art static hash table. For more
information, please check our paper:

[**Better GPU Hash Tables**](https://arxiv.org/abs/2108.07232)\
*[Muhammad A. Awad](https://maawad.github.io/), [Saman
Ashkiani](https://scholar.google.com/citations?user=Z4_ZfiEAAAAJ&hl=en),
[Serban D. Porumbescu](https://web.cs.ucdavis.edu/~porumbes/), [Martín
Farach-Colton](https://people.cs.rutgers.edu/~farach/), and [John D.
Owens](https://www.ece.ucdavis.edu/~jowens/)*

# []{#autotoc_md1 .anchor} Key features

-   State-of-the-art static GPU hash tables
-   Device and host side APIs
-   Support for different types of keys and values
-   Standard-like APIs

# []{#autotoc_md2 .anchor} How to use

BGHT is a header-only library. To use the library, you can add it as a
submodule or use [CMake Package Manager
(CPM)](https://github.com/cpm-cmake/CPM.cmake) to fetch the library into
your CMake-based project ([complete
example](https://github.com/owensgroup/BGHT/tree/main/test/cpm)).

::: fragment
::: line
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
:::

::: line
CPMAddPackage(
:::

::: line
NAME bght
:::

::: line
GITHUB_REPOSITORY owensgroup/BGHT
:::

::: line
GIT_TAG main
:::

::: line
OPTIONS
:::

::: line
\"build_tests OFF\"
:::

::: line
\"build_benchmarks OFF\"
:::

::: line
)
:::

::: line
target_link_libraries(my_library PRIVATE bght)
:::
:::

## []{#autotoc_md3 .anchor} APIs

All the data structures follow the C++ standard hash map
(`std::unordered_map`) APIs closely. An example APIs for BCHT is shown
below:

::: fragment
::: line
{c++}
:::

::: line
template \<class Key,
:::

::: line
class T,
:::

::: line
class Hash = bght::universal_hash\<Key\>,
:::

::: line
class KeyEqual = bght::equal_to\<Key\>,
:::

::: line
cuda::thread_scope Scope = cuda::thread_scope_device,
:::

::: line
class Allocator = bght::cuda_allocator\<char\>,
:::

::: line
int B = 16\> class bcht;
:::
:::

### []{#autotoc_md4 .anchor} Member functions

::: fragment
::: line
{c++}
:::

::: line
// Constructor
:::

::: line
bcht(std::size_t capacity,
:::

::: line
Key sentinel_key,
:::

::: line
T sentinel_value,
:::

::: line
Allocator const& allocator = Allocator{});
:::

::: line
// Host-side APIs
:::

::: line
template \<typename InputIt\>
:::

::: line
bool insert(InputIt first, InputIt last, cudaStream_t stream = 0);
:::

::: line
template \<typename InputIt, typename OutputIt\>
:::

::: line
void find(InputIt first, InputIt last, OutputIt output_begin,
cudaStream_t stream = 0);
:::

::: line
// Device-side APIs
:::

::: line
template \<typename tile_type\>
:::

::: line
\_\_device\_\_ bool insert(value_type const& pair, tile_type const&
tile);
:::

::: line
template \<typename tile_type\>
:::

::: line
\_\_device\_\_ mapped_type find(key_type const& key, tile_type const&
tile);
:::
:::

## []{#autotoc_md5 .anchor} Member types

::: fragment
::: line
Member type Definition
:::

::: line
key_type Key
:::

::: line
mapped_type T
:::

::: line
value_type bght::pair\<Key, T\>
:::

::: line
allocator_type Allocator
:::

::: line
bucket_size Bucket size for device-side APIs cooperative groups tile
construction
:::
:::

### []{#autotoc_md6 .anchor} Example

::: fragment
::: line
{c++}
:::

::: line
// Example using host-side APIs
:::

::: line
#include \<bcht.hpp\>
:::

::: line
int main(){
:::

::: line
using key_type = uint32_t;
:::

::: line
using value_type = uint32_t;
:::

::: line
using pair_type = bght::pair\<key_type, value_type\>;
:::

::: line
std::size_t capacity = 128; std::size_t num_keys = 64;
:::

::: line
key_type invalid_key = 0; value_type invalid_value = 0; // sentinel key
and value
:::

::: line
:::

::: line
bght::bcht\<key_type, value_type\> table(capacity, invalid_key,
invalid_value); //ctor
:::

::: line
:::

::: line
pair_type\* pairs; // input pairs
:::

::: line
// \... allocate pairs
:::

::: line
:::

::: line
bool success = table.insert(pairs, pairs + num_keys);
:::

::: line
assert(success);
:::

::: line
:::

::: line
key_type\* queries; // query keys
:::

::: line
value_type\* results; // query result
:::

::: line
// \... allocate queries and results
:::

::: line
table.find(queries, queries + num_keys, results);
:::

::: line
}
:::
:::

::: fragment
::: line
{c++}
:::

::: line
// Example using device-side APIs
:::

::: line
template\<class HashMap\>
:::

::: line
\_\_global\_\_ void kernel(HashMap table){
:::

::: line
// construct tile
:::

::: line
auto block = cooperative_groups::this_thread_block();
:::

::: line
auto tile =
cooperative_groups::tiled_partition\<HashMap::bucket_size\>(block);
:::

::: line
pair_type pair{\...};
:::

::: line
table.insert(pair, tile);
:::

::: line
pair_type query{..};
:::

::: line
query.second = talbe.find(query.first, tile);
:::

::: line
}
:::

::: line
int main(){
:::

::: line
// Call the hash table constructor on the CPU
:::

::: line
bght::bcht\<key_type, value_type\> table(\...);
:::

::: line
// Pass the hash table to a GPU kernel
:::

::: line
kernel\<\<\<\...\>\>\>(table);
:::

::: line
}
:::
:::

# []{#autotoc_md7 .anchor} Requirements and limitations

Please create an issue if you face challenges with any of the following
limitations and requirements.

## []{#autotoc_md8 .anchor} Requirements

-   C++17/CUDA C++17
-   NVIDIA Volta GPU or later microarchitectures
-   CMake 3.8 or later
-   CUDA 11.5 or later

## []{#autotoc_md9 .anchor} limitations

-   Currently hash tables based on cuckoo hashing do not support
    concurrent insertion and queries. IHT and P2BHT support concurrent
    insertions and queries.
-   Keys must be unique
-   Construction of the data structures offered *may* fail. In these
    scenarios, reconstructing the table using a larger capacity or a
    lower load factor should be considered. Our paper offers recommended
    hash table load factors (for uniformly distributed unsigned keys) to
    achieve at least a 99% success rate ([See Fig.
    2](https://arxiv.org/abs/2108.07232)). For example, BCHT will offer
    a 100% success rate for up to 0.991 load factor. Please create an
    issue if you encounter any problems with different key
    distributions.

# []{#autotoc_md10 .anchor} Reproducing the arXiv paper results

To reproduce the results, follow the following
[steps](md_reproduce.html){.el}. You can also view our results
[here](md_results.html){.el}. If you find any mismatch (either faster or
slower) between the results offered in the repository or the paper,
please create an issue, and we will investigate the performance changes.

# []{#autotoc_md11 .anchor} Benchmarks

Please check our [paper](https://arxiv.org/abs/2108.07232) for
comprehensive analysis and benchmarks. Also, see the following steps to
[reproduce](md_reproduce.html){.el} the results.

An additional comparison of our BCHT to `cucCollection`\'s
`cuco::static_map` is shown below. The comparison is between our BCHT
with B = 16 (default configuration) and `cuco::static_map`. Input keys
(50 million pairs) are uniformly distributed unsigned keys. The
benchmarking was performed on an NVIDIA Titan V GPU (higher is better):

![](/figs/arxiv/NVIDIA-TITAN-V/bcht_vs_cuco.svg){.inline
style="pointer-events: none;"}

# []{#autotoc_md12 .anchor} Questions and bug report

Please create an issue. We will welcome any contributions that improve
the usability and quality of our repository.

# []{#autotoc_md13 .anchor} Bibtex

::: fragment
::: line
\@article{Awad:2021:BGH,
:::

::: line
title = {Better {GPU} Hash Tables},
:::

::: line
author = {Muhammad A. Awad and Saman Ashkiani and Serban D.
:::

::: line
Porumbescu and Mart{\\\'{i}}n Farach-Colton and John
:::

::: line
D. Owens},
:::

::: line
year = 2021,
:::

::: line
month = aug,
:::

::: line
primaryclass = {cs.DS},
:::

::: line
journal = {CoRR},
:::

::: line
volume = {abs/2108.07232},
:::

::: line
archiveprefix = {arXiv},
:::

::: line
number = {2108.07232},
:::

::: line
eprint = {2108.07232},
:::

::: line
nonrefereed = {true},
:::

::: line
code = {https://github.com/owensgroup/BGHT}
:::

::: line
}
:::
:::

# []{#autotoc_md14 .anchor} Acknowledgments

The structure and organization of the repository were inspired by
[NVIDIA\'s cuCollection](https://github.com/nviDIA/cuCollections/) and
[RXMesh](https://github.com/owensgroup/RXMesh).
:::
:::

</div>

------------------------------------------------------------------------

[Generated by [![doxygen](doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
