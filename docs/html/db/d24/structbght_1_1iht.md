::: {#top}
::: {#titlearea}
+-----------------------------------------------------------------------+
| ::: {#projectname}                                                    |
| BGHT                                                                  |
| :::                                                                   |
|                                                                       |
| ::: {#projectbrief}                                                   |
| High-performance static GPU hash tables.                              |
| :::                                                                   |
+-----------------------------------------------------------------------+
:::

::: {#main-nav}
:::

::: {#MSearchSelectWindow onmouseover="return searchBox.OnSearchSelectShow()" onmouseout="return searchBox.OnSearchSelectHide()" onkeydown="return searchBox.OnSearchSelectKey(event)"}
:::

::: {#MSearchResultsWindow}
:::

::: {#nav-path .navpath}
-   **bght**
-   [iht](../../db/d24/structbght_1_1iht.html){.el}
:::
:::

::: header
::: summary
[Public Types](#pub-types) \| [Public Member Functions](#pub-methods) \|
[Static Public Attributes](#pub-static-attribs)
:::

::: headertitle
::: title
bght::iht\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>
Struct Template Reference
:::
:::
:::

::: contents
IHT IHT (iceberg hash table) is an associative static GPU hash table
that contains key-value pairs with unique keys. The hash table is an
open addressing hash table based on the cuckoo hashing probing scheme
(bucketed and using a primary hash function and two secondary hash
functions). [More\...](../../db/d24/structbght_1_1iht.html#details)

`#include <iht.hpp>`

+-----------------------------------+-----------------------------------+
| ## []{#pub-types} Public T        |                                   |
| ypes {#public-types .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **value_type** = pair\< Key, T \> |
| 148549cd4c6e47575bfc9cd963ded682} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_type** = Key                |
| 06bd24c9fed560f38c7f6fdbc5002cc6} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **mapped_type** = T               |
| 7ac5c89864515b4606da165f105a7e7a} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_type** =            |
| 2934a2b1047386281092b746263581fa} | cuda::atomic\< value_type, Scope  |
| using                             | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **allocator_type** = Allocator    |
| f49723f71ff612171599b8c897174181} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **hasher** = Hash                 |
| 8976f63021c5d62fac63092a97ed7b13} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **size_type** = std::size_t       |
| e7c271794f330b9b57b1d9a286ea2515} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_allocator_type** =  |
| c7d1c32e54a49c46d588662055f601d9} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | atomic_pair_type \>               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **pool_allocator_type** =         |
| 7ba989411ff13a8be0d76a6850927799} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\< bool |
|                                   | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **size_type_allocator_type** =    |
| 9564ec8edeed7fe488a7fe69a74946c3} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | size_type \>                      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_equal** = KeyEqual          |
| 32e2f61403ade32d10acb42ef9ffae0a} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-method                 |                                   |
| s} Public Member Functions {#publ |                                   |
| ic-member-functions .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
|                                   | [iht](../../db/                   |
|                                   | d24/structbght_1_1iht.html#a02e2c |
|                                   | 454d005b7cf91363feb1477c7b9){.el} |
|                                   | (std::size_t capacity, Key        |
|                                   | sentinel_key, T sentinel_value,   |
|                                   | Allocator const                   |
|                                   | &allocator=Allocator{})           |
+-----------------------------------+-----------------------------------+
|                                   | Constructs the hash table with    |
|                                   | the specified capacity and uses   |
|                                   | the specified sentinel key and    |
|                                   | value to define a sentinel pair.  |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#a0 |
|                                   | 2e2c454d005b7cf91363feb1477c7b9)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **iht** (const                    |
| 725ba654f985af3bdbb95830f8e26776} | [iht](../../db                    |
|                                   | /d24/structbght_1_1iht.html){.el} |
|                                   | &other)                           |
+-----------------------------------+-----------------------------------+
|                                   | A shallow-copy constructor.\      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **iht**                           |
| 46dd032710006cc6cffcdc62a8da6564} | ([iht](../../db                   |
|                                   | /d24/structbght_1_1iht.html){.el} |
|                                   | &&)=delete                        |
+-----------------------------------+-----------------------------------+
|                                   | Move constructor is currently     |
|                                   | deleted.\                         |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=** (const              |
| 9299d3e97a061b2319ccbfcc4d36fe71} | [iht](../../db                    |
| [iht](../../db                    | /d24/structbght_1_1iht.html){.el} |
| /d24/structbght_1_1iht.html){.el} | &)=delete                         |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The assignment operator is        |
|                                   | currently deleted.\               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=**                     |
| 93b6e68ec5e324e1aae55cbbb900e24e} | ([iht](../../db                   |
| [iht](../../db                    | /d24/structbght_1_1iht.html){.el} |
| /d24/structbght_1_1iht.html){.el} | &&)=delete                        |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The move assignment operator is   |
|                                   | currently deleted.\               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **\~iht** ()                      |
| 1eb1c14a9683be082405b20cd4ac0fb5} |                                   |
|                                   |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Destructor that destroys the hash |
|                                   | map and deallocate memory if no   |
|                                   | copies exist.\                    |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **clear** ()                      |
| 076471ca81f32b45a0d91ac5a004946e} |                                   |
| void                              |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Clears the hash map and resets    |
|                                   | all slots.\                       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt \>     |                                   |
+-----------------------------------+-----------------------------------+
| bool                              | [insert](../../db/                |
|                                   | d24/structbght_1_1iht.html#afcb16 |
|                                   | a7c2cc72a322d90e0bd8805cb50){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for inserting all   |
|                                   | pairs defined by the input        |
|                                   | argument iterators. All keys in   |
|                                   | the range must be unique and must |
|                                   | not exist in the hash table.      |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#af |
|                                   | cb16a7c2cc72a322d90e0bd8805cb50)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt ,      |                                   |
| typename OutputIt \>              |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [find](../../db/                  |
|                                   | d24/structbght_1_1iht.html#ac6aa7 |
|                                   | 1eb51d1f76879cd0ec3c398fadc){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | OutputIt output_begin,            |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for finding all     |
|                                   | keys defined by the input         |
|                                   | argument iterators.               |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#ac |
|                                   | 6aa71eb51d1f76879cd0ec3c398fadc)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename tile_type \>   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ bool               | [insert](../../db/                |
|                                   | d24/structbght_1_1iht.html#abade7 |
|                                   | d2b541a2718bbae41be40a0ae40){.el} |
|                                   | (value_type const &pair,          |
|                                   | tile_type const &tile)            |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative insertion |
|                                   | API that inserts a single pair    |
|                                   | into the hash map.                |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#ab |
|                                   | ade7d2b541a2718bbae41be40a0ae40)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename tile_type \>   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ mapped_type        | [find](../../db/                  |
|                                   | d24/structbght_1_1iht.html#a5f0c2 |
|                                   | 22a207ee2050f6910a33a312b5d){.el} |
|                                   | (key_type const &key, tile_type   |
|                                   | const &tile)                      |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative find API  |
|                                   | that finds a single pair into the |
|                                   | hash map.                         |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#a5 |
|                                   | f0c222a207ee2050f6910a33a312b5d)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename RNG \>         |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [ra                               |
|                                   | ndomize_hash_functions](../../db/ |
|                                   | d24/structbght_1_1iht.html#acf9f5 |
|                                   | c49d7306e3567482b5a26b3b88b){.el} |
|                                   | (RNG &rng)                        |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API to randomize the    |
|                                   | hash functions used for the       |
|                                   | probing scheme. This can be used  |
|                                   | when the hash table construction  |
|                                   | fails. The hash table must be     |
|                                   | cleared after a call to this      |
|                                   | function.                         |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#ac |
|                                   | f9f5c49d7306e3567482b5a26b3b88b)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| size_type                         | [size](../../db/                  |
|                                   | d24/structbght_1_1iht.html#a0efa9 |
|                                   | 40784530baf1893c7cfbf901651){.el} |
|                                   | (cudaStream_t stream=0)           |
+-----------------------------------+-----------------------------------+
|                                   | Compute the number of elements in |
|                                   | the map.                          |
|                                   | [More\...](../..                  |
|                                   | /db/d24/structbght_1_1iht.html#a0 |
|                                   | efa940784530baf1893c7cfbf901651)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-static-attribs}        |                                   |
|  Static Public Attributes {#stati |                                   |
| c-public-attributes .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **bucket_size** = B               |
| 82a62aa7555d598da596076c33c1cc61} |                                   |
| static constexpr auto             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

[]{#details}

## Detailed Description {#detailed-description .groupheader}

::: textblock
::: compoundTemplParams
template\<class Key, class T, class Hash = bght::universal_hash\<Key\>,
class KeyEqual = bght::equal_to\<Key\>, cuda::thread_scope Scope =
cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>\
struct bght::iht\< Key, T, Hash, KeyEqual, Scope, Allocator, B,
Threshold \>
:::

IHT IHT (iceberg hash table) is an associative static GPU hash table
that contains key-value pairs with unique keys. The hash table is an
open addressing hash table based on the cuckoo hashing probing scheme
(bucketed and using a primary hash function and two secondary hash
functions).

Template Parameters

:   ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Key         Type for the hash map key
      T           Type for the mapped value
      Hash        Unary function object class that defines the hash function. The function must have an `initialize_hf` specialization to initialize the hash function using a random number generator
      KeyEqual    Binary function object class that compares two keys
      Allocator   The allocator to use for allocating GPU device memory
      B           Bucket size for the hash table
      Threshold   Iceberg threshold
      ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

## Constructor & Destructor Documentation {#constructor-destructor-documentation .groupheader}

[]{#a02e2c454d005b7cf91363feb1477c7b9}

## [[◆ ](#a02e2c454d005b7cf91363feb1477c7b9)]{.permalink}iht() {#iht .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ------------------------------
  [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::[iht](../../db/d24/structbght_1_1iht.html){.el}   (   std::size_t          *capacity*,
                                                                                                                                                                           Key                  *sentinel_key*,
                                                                                                                                                                           T                    *sentinel_value*,
                                                                                                                                                                           Allocator const &    *allocator* = `Allocator{}` 
                                                                                                                                                                       )                        
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ------------------------------
:::

::: memdoc
Constructs the hash table with the specified capacity and uses the
specified sentinel key and value to define a sentinel pair.

Parameters

:   ---------------- ----------------------------------------------------------------------------------------------------------------------
      capacity         The number of slots to use in the hash table. If the capacity is not multiple of the bucket size, it will be rounded
      sentinel_key     A reserved sentinel key that defines an empty key
      sentinel_value   A reserved sentinel value that defines an empty value
      allocator        The allocator to use for allocating GPU device memory
      ---------------- ----------------------------------------------------------------------------------------------------------------------
:::
:::

## Member Function Documentation {#member-function-documentation .groupheader}

[]{#ac6aa71eb51d1f76879cd0ec3c398fadc}

## [[◆ ](#ac6aa71eb51d1f76879cd0ec3c398fadc)]{.permalink}find() [\[1/2\]]{.overload} {#find-12 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

::: memtemplate
template\<typename InputIt , typename OutputIt \>
:::

  ------------------------------------------------------------------------------------------------------------------------------ --- --------------- -----------------
  void [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::find   (   InputIt         *first*,
                                                                                                                                     InputIt         *last*,
                                                                                                                                     OutputIt        *output_begin*,
                                                                                                                                     cudaStream_t    *stream* = `0` 
                                                                                                                                 )                   
  ------------------------------------------------------------------------------------------------------------------------------ --- --------------- -----------------
:::

::: memdoc
Host-side API for finding all keys defined by the input argument
iterators.

Template Parameters

:   ---------- -------------------------------------------------------------
      InputIt    Device-side iterator that can be converted to `key_type`
      OutputIt   Device-side iterator that can be converted to `mapped_type`
      ---------- -------------------------------------------------------------

```{=html}
<!-- -->
```

Parameters

:   -------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      first          An iterator defining the beginning of the input keys to find
      last           An iterator defining the end of the input keys to find
      output_begin   An iterator defining the beginning of the output buffer to store the results into. The size of the buffer must match the number of queries defined by the input iterators.
      stream         A CUDA stream where the insertion operation will take place
      -------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::
:::

[]{#a5f0c222a207ee2050f6910a33a312b5d}

## [[◆ ](#a5f0c222a207ee2050f6910a33a312b5d)]{.permalink}find() [\[2/2\]]{.overload} {#find-22 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

::: memtemplate
template\<typename tile_type \>
:::

  ---------------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ---------
  \_\_device\_\_ mapped_type [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::find   (   key_type const &     *key*,
                                                                                                                                                           tile_type const &    *tile* 
                                                                                                                                                       )                        
  ---------------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ---------
:::

::: memdoc
Device-side cooperative find API that finds a single pair into the hash
map.

Template Parameters

:   ----------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      tile_type   A cooperative group tile with a size that must match the bucket size of the hash map (i.e., `bucket_size`). It must support the tile-wide intrinsics `ballot`, `shfl`
      ----------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

```{=html}
<!-- -->
```

Parameters

:   ------ -------------------------------------------------------------------------------------------------------
      key    A key to find in the hash map. The key must be the same for all threads in the cooperative group tile
      tile   The cooperative group tile
      ------ -------------------------------------------------------------------------------------------------------

```{=html}
<!-- -->
```

Returns
:   The value of the key if it exists in the map or the `sentinel_value`
    if the key does not exist in the hash map
:::
:::

[]{#afcb16a7c2cc72a322d90e0bd8805cb50}

## [[◆ ](#afcb16a7c2cc72a322d90e0bd8805cb50)]{.permalink}insert() [\[1/2\]]{.overload} {#insert-12 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

::: memtemplate
template\<typename InputIt \>
:::

  -------------------------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
  bool [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::insert   (   InputIt         *first*,
                                                                                                                                       InputIt         *last*,
                                                                                                                                       cudaStream_t    *stream* = `0` 
                                                                                                                                   )                   
  -------------------------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
:::

::: memdoc
Host-side API for inserting all pairs defined by the input argument
iterators. All keys in the range must be unique and must not exist in
the hash table.

Template Parameters

:   --------- -------------------------------------------------------------
      InputIt   Device-side iterator that can be converted to `value_type`.
      --------- -------------------------------------------------------------

```{=html}
<!-- -->
```

Parameters

:   -------- -----------------------------------------------------------------
      first    An iterator defining the beginning of the input pairs to insert
      last     An iterator defining the end of the input pairs to insert
      stream   A CUDA stream where the insertion operation will take place
      -------- -----------------------------------------------------------------

```{=html}
<!-- -->
```

Returns
:   A boolean indicating success (true) or failure (false) of the
    insertion operation.
:::
:::

[]{#abade7d2b541a2718bbae41be40a0ae40}

## [[◆ ](#abade7d2b541a2718bbae41be40a0ae40)]{.permalink}insert() [\[2/2\]]{.overload} {#insert-22 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

::: memtemplate
template\<typename tile_type \>
:::

  ----------------------------------------------------------------------------------------------------------------------------------------------- --- --------------------- ---------
  \_\_device\_\_ bool [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::insert   (   value_type const &    *pair*,
                                                                                                                                                      tile_type const &     *tile* 
                                                                                                                                                  )                         
  ----------------------------------------------------------------------------------------------------------------------------------------------- --- --------------------- ---------
:::

::: memdoc
Device-side cooperative insertion API that inserts a single pair into
the hash map.

Template Parameters

:   ----------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      tile_type   A cooperative group tile with a size that must match the bucket size of the hash map (i.e., `bucket_size`). It must support the tile-wide intrinsics `ballot`, `shfl`
      ----------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

```{=html}
<!-- -->
```

Parameters

:   ------ -----------------------------------------------------------------------------------------------------------------------
      pair   A key-value pair to insert into the hash map. The pair must be the same for all threads in the cooperative group tile
      tile   The cooperative group tile
      ------ -----------------------------------------------------------------------------------------------------------------------

```{=html}
<!-- -->
```

Returns
:   A boolean indicating success (true) or failure (false) of the
    insertion operation.
:::
:::

[]{#acf9f5c49d7306e3567482b5a26b3b88b}

## [[◆ ](#acf9f5c49d7306e3567482b5a26b3b88b)]{.permalink}randomize_hash_functions() {#randomize_hash_functions .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

::: memtemplate
template\<typename RNG \>
:::

  -------------------------------------------------------------------------------------------------------------------------------------------------- --- -------- ------- --- --
  void [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::randomize_hash_functions   (   RNG &    *rng*   )   
  -------------------------------------------------------------------------------------------------------------------------------------------------- --- -------- ------- --- --
:::

::: memdoc
Host-side API to randomize the hash functions used for the probing
scheme. This can be used when the hash table construction fails. The
hash table must be cleared after a call to this function.

Template Parameters

:   ----- ----------------------------------
      RNG   A pseudo-random number generator
      ----- ----------------------------------

```{=html}
<!-- -->
```

Parameters

:   ----- --------------------------------------------------------
      rng   An instantiation of the pseudo-random number generator
      ----- --------------------------------------------------------
:::
:::

[]{#a0efa940784530baf1893c7cfbf901651}

## [[◆ ](#a0efa940784530baf1893c7cfbf901651)]{.permalink}size() {#size .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16, int Threshold = 14\>
:::

  ----------------------------------------------------------------------------------------------------------------------------------- --- --------------- ---------------- --- --
  size_type [bght::iht](../../db/d24/structbght_1_1iht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B, Threshold \>::size   (   cudaStream_t    *stream* = `0`   )   
  ----------------------------------------------------------------------------------------------------------------------------------- --- --------------- ---------------- --- --
:::

::: memdoc
Compute the number of elements in the map.

Returns
:   The number of elements in the map
:::
:::

------------------------------------------------------------------------

The documentation for this struct was generated from the following file:

-   include/[iht.hpp](../../d1/db6/iht_8hpp_source.html){.el}
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
