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
-   [p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}
:::
:::

::: header
::: summary
[Public Types](#pub-types) \| [Public Member Functions](#pub-methods) \|
[Static Public Attributes](#pub-static-attribs)
:::

::: headertitle
::: title
bght::p2bht\< Key, T, Hash, KeyEqual, Scope, Allocator, B \> Struct
Template Reference
:::
:::
:::

::: contents
P2BHT P2BHT (power-of-two bucketed hash table) is an associative static
GPU hash table that contains key-value pairs with unique keys. The hash
table is an open addressing hash table based on the power-of-two hashing
to balance loads between buckets (bucketed and using two hash
functions). [More\...](../../d6/dcc/structbght_1_1p2bht.html#details)

`#include <p2bht.hpp>`

+-----------------------------------+-----------------------------------+
| ## []{#pub-types} Public T        |                                   |
| ypes {#public-types .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **value_type** = pair\< Key, T \> |
| e4e61f0ba7d9a409a3c7b4b2551cf0d5} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_type** = Key                |
| 59f17bb3a17d9172bea0dd5a88a6e079} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **mapped_type** = T               |
| 14a5a2b53bb0f053121234ffdc22fe60} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_type** =            |
| dcca84496f179da5ea46526caed410f0} | cuda::atomic\< value_type, Scope  |
| using                             | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **allocator_type** = Allocator    |
| 40458c645c120911ae1f7f75bebfc336} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **hasher** = Hash                 |
| bc70d74d615698fe02cb2bf1b074efa4} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **size_type** = std::size_t       |
| 122b6c43c72732daff300abd04dd7550} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_allocator_type** =  |
| 5a8c238487b240303a0458df07265c3f} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | atomic_pair_type \>               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **pool_allocator_type** =         |
| cc3a0f02b65c193b985d36fee15123fd} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\< bool |
|                                   | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **size_type_allocator_type** =    |
| b060495e523efae36b5531bc66d4a441} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | size_type \>                      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_equal** = KeyEqual          |
| 137fb4620186537b75aaf75067f205c6} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-method                 |                                   |
| s} Public Member Functions {#publ |                                   |
| ic-member-functions .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
|                                   | [p2bht](../../d6/dc               |
|                                   | c/structbght_1_1p2bht.html#a9e57c |
|                                   | de3bed4b452a1699802f1390b58){.el} |
|                                   | (std::size_t capacity, Key        |
|                                   | sentinel_key, T sentinel_value,   |
|                                   | Allocator const                   |
|                                   | &allocator=Allocator{})           |
+-----------------------------------+-----------------------------------+
|                                   | Constructs the hash table with    |
|                                   | the specified capacity and uses   |
|                                   | the specified sentinel key and    |
|                                   | value to define a sentinel pair.  |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#a9 |
|                                   | e57cde3bed4b452a1699802f1390b58)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **p2bht** (const                  |
| 5b86e49577c0fec06aed5c6990cf93b9} | [p2bht](../../d6/d                |
|                                   | cc/structbght_1_1p2bht.html){.el} |
|                                   | &other)                           |
+-----------------------------------+-----------------------------------+
|                                   | A shallow-copy constructor.\      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **p2bht**                         |
| fd4542b3692395f6af3ef6a214fc0001} | ([p2bht](../../d6/d               |
|                                   | cc/structbght_1_1p2bht.html){.el} |
|                                   | &&)=delete                        |
+-----------------------------------+-----------------------------------+
|                                   | Move constructor is currently     |
|                                   | deleted.\                         |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=** (const              |
| 447e9bbf421c9d6a8983da989629994d} | [p2bht](../../d6/d                |
| [p2bht](../../d6/d                | cc/structbght_1_1p2bht.html){.el} |
| cc/structbght_1_1p2bht.html){.el} | &)=delete                         |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The assignment operator is        |
|                                   | currently deleted.\               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=**                     |
| f71794cec02a46adc02a875ca2048899} | ([p2bht](../../d6/d               |
| [p2bht](../../d6/d                | cc/structbght_1_1p2bht.html){.el} |
| cc/structbght_1_1p2bht.html){.el} | &&)=delete                        |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The move assignment operator is   |
|                                   | currently deleted.\               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **\~p2bht** ()                    |
| ee3fe77f75a7825a060d9b891726cff3} |                                   |
|                                   |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Destructor that destroys the hash |
|                                   | map and deallocate memory if no   |
|                                   | copies exist.\                    |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **clear** ()                      |
| b2181363291624ae8b152a4438ff6caf} |                                   |
| void                              |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Clears the hash map and resets    |
|                                   | all slots.\                       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt \>     |                                   |
+-----------------------------------+-----------------------------------+
| bool                              | [insert](../../d6/dc              |
|                                   | c/structbght_1_1p2bht.html#afaf41 |
|                                   | c013493f4912537f20ec716f015){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for inserting all   |
|                                   | pairs defined by the input        |
|                                   | argument iterators. All keys in   |
|                                   | the range must be unique and must |
|                                   | not exist in the hash table.      |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#af |
|                                   | af41c013493f4912537f20ec716f015)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt ,      |                                   |
| typename OutputIt \>              |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [find](../../d6/dc                |
|                                   | c/structbght_1_1p2bht.html#a1d024 |
|                                   | b93ef3392e9634b1160ab679d92){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | OutputIt output_begin,            |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for finding all     |
|                                   | keys defined by the input         |
|                                   | argument iterators.               |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#a1 |
|                                   | d024b93ef3392e9634b1160ab679d92)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename tile_type \>   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ bool               | [insert](../../d6/dc              |
|                                   | c/structbght_1_1p2bht.html#ad6bcc |
|                                   | a773f08880cdfe156352b8925d8){.el} |
|                                   | (value_type const &pair,          |
|                                   | tile_type const &tile)            |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative insertion |
|                                   | API that inserts a single pair    |
|                                   | into the hash map.                |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#ad |
|                                   | 6bcca773f08880cdfe156352b8925d8)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename tile_type \>   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ mapped_type        | [find](../../d6/dc                |
|                                   | c/structbght_1_1p2bht.html#a913cc |
|                                   | a5c95d7b9c7ad5a61ea6769e9d6){.el} |
|                                   | (key_type const &key, tile_type   |
|                                   | const &tile)                      |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative find API  |
|                                   | that finds a single pair into the |
|                                   | hash map.                         |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#a9 |
|                                   | 13cca5c95d7b9c7ad5a61ea6769e9d6)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename RNG \>         |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [rand                             |
|                                   | omize_hash_functions](../../d6/dc |
|                                   | c/structbght_1_1p2bht.html#a9112b |
|                                   | 518945f384f948ed52aa0c5ad80){.el} |
|                                   | (RNG &rng)                        |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API to randomize the    |
|                                   | hash functions used for the       |
|                                   | probing scheme. This can be used  |
|                                   | when the hash table construction  |
|                                   | fails. The hash table must be     |
|                                   | cleared after a call to this      |
|                                   | function.                         |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#a9 |
|                                   | 112b518945f384f948ed52aa0c5ad80)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| size_type                         | [size](../../d6/dc                |
|                                   | c/structbght_1_1p2bht.html#a4f95a |
|                                   | 1f7a39e693a0ca1cd67951c1ec7){.el} |
|                                   | (cudaStream_t stream=0)           |
+-----------------------------------+-----------------------------------+
|                                   | Compute the number of elements in |
|                                   | the map.                          |
|                                   | [More\...](../../d                |
|                                   | 6/dcc/structbght_1_1p2bht.html#a4 |
|                                   | f95a1f7a39e693a0ca1cd67951c1ec7)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-static-attribs}        |                                   |
|  Static Public Attributes {#stati |                                   |
| c-public-attributes .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **bucket_size** = B               |
| fccd5329f7841fb32ca3018f1cf851c0} |                                   |
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
bght::cuda_allocator\<char\>, int B = 16\>\
struct bght::p2bht\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>
:::

P2BHT P2BHT (power-of-two bucketed hash table) is an associative static
GPU hash table that contains key-value pairs with unique keys. The hash
table is an open addressing hash table based on the power-of-two hashing
to balance loads between buckets (bucketed and using two hash
functions).

Template Parameters

:   ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Key         Type for the hash map key
      T           Type for the mapped value
      Hash        Unary function object class that defines the hash function. The function must have an `initialize_hf` specialization to initialize the hash function using a random number generator
      KeyEqual    Binary function object class that compares two keys
      Allocator   The allocator to use for allocating GPU device memory
      B           Bucket size for the hash table
      ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

## Constructor & Destructor Documentation {#constructor-destructor-documentation .groupheader}

[]{#a9e57cde3bed4b452a1699802f1390b58}

## [[◆ ](#a9e57cde3bed4b452a1699802f1390b58)]{.permalink}p2bht() {#p2bht .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ------------------------------
  [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::[p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}   (   std::size_t          *capacity*,
                                                                                                                                                                        Key                  *sentinel_key*,
                                                                                                                                                                        T                    *sentinel_value*,
                                                                                                                                                                        Allocator const &    *allocator* = `Allocator{}` 
                                                                                                                                                                    )                        
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ------------------------------
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

[]{#a1d024b93ef3392e9634b1160ab679d92}

## [[◆ ](#a1d024b93ef3392e9634b1160ab679d92)]{.permalink}find() [\[1/2\]]{.overload} {#find-12 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

::: memtemplate
template\<typename InputIt , typename OutputIt \>
:::

  ----------------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
  void [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::find   (   InputIt         *first*,
                                                                                                                              InputIt         *last*,
                                                                                                                              OutputIt        *output_begin*,
                                                                                                                              cudaStream_t    *stream* = `0` 
                                                                                                                          )                   
  ----------------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
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

[]{#a913cca5c95d7b9c7ad5a61ea6769e9d6}

## [[◆ ](#a913cca5c95d7b9c7ad5a61ea6769e9d6)]{.permalink}find() [\[2/2\]]{.overload} {#find-22 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

::: memtemplate
template\<typename tile_type \>
:::

  --------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ---------
  \_\_device\_\_ mapped_type [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::find   (   key_type const &     *key*,
                                                                                                                                                    tile_type const &    *tile* 
                                                                                                                                                )                        
  --------------------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ---------
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

[]{#afaf41c013493f4912537f20ec716f015}

## [[◆ ](#afaf41c013493f4912537f20ec716f015)]{.permalink}insert() [\[1/2\]]{.overload} {#insert-12 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

::: memtemplate
template\<typename InputIt \>
:::

  ------------------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
  bool [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::insert   (   InputIt         *first*,
                                                                                                                                InputIt         *last*,
                                                                                                                                cudaStream_t    *stream* = `0` 
                                                                                                                            )                   
  ------------------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
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

[]{#ad6bcca773f08880cdfe156352b8925d8}

## [[◆ ](#ad6bcca773f08880cdfe156352b8925d8)]{.permalink}insert() [\[2/2\]]{.overload} {#insert-22 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

::: memtemplate
template\<typename tile_type \>
:::

  ---------------------------------------------------------------------------------------------------------------------------------------- --- --------------------- ---------
  \_\_device\_\_ bool [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::insert   (   value_type const &    *pair*,
                                                                                                                                               tile_type const &     *tile* 
                                                                                                                                           )                         
  ---------------------------------------------------------------------------------------------------------------------------------------- --- --------------------- ---------
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

[]{#a9112b518945f384f948ed52aa0c5ad80}

## [[◆ ](#a9112b518945f384f948ed52aa0c5ad80)]{.permalink}randomize_hash_functions() {#randomize_hash_functions .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

::: memtemplate
template\<typename RNG \>
:::

  ------------------------------------------------------------------------------------------------------------------------------------------- --- -------- ------- --- --
  void [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::randomize_hash_functions   (   RNG &    *rng*   )   
  ------------------------------------------------------------------------------------------------------------------------------------------- --- -------- ------- --- --
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

[]{#a4f95a1f7a39e693a0ca1cd67951c1ec7}

## [[◆ ](#a4f95a1f7a39e693a0ca1cd67951c1ec7)]{.permalink}size() {#size .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

  ---------------------------------------------------------------------------------------------------------------------------- --- --------------- ---------------- --- --
  size_type [bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::size   (   cudaStream_t    *stream* = `0`   )   
  ---------------------------------------------------------------------------------------------------------------------------- --- --------------- ---------------- --- --
:::

::: memdoc
Compute the number of elements in the map.

Returns
:   The number of elements in the map
:::
:::

------------------------------------------------------------------------

The documentation for this struct was generated from the following file:

-   include/[p2bht.hpp](../../de/dbb/p2bht_8hpp_source.html){.el}
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
