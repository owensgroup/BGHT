::: {#top}
::: {#titlearea}
+-----------------------------------------------------------------------+
| ::: {#projectname}                                                    |
| My Project                                                            |
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
-   [bcht](structbght_1_1bcht.html){.el}
:::
:::

::: header
::: summary
[Public Types](#pub-types) \| [Public Member Functions](#pub-methods) \|
[Static Public Attributes](#pub-static-attribs) \| [Friends](#friends)
\| [List of all members](structbght_1_1bcht-members.html)
:::

::: headertitle
::: title
bght::bcht\< Key, T, Hash, KeyEqual, Scope, Allocator, B \> Struct
Template Reference
:::
:::
:::

::: contents
BCHT BCHT (bucketed cuckoo hash table) is an associative static GPU hash
table that contains key-value pairs with unique keys. The hash table is
an open addressing hash table based on the cuckoo hashing probing scheme
(bucketed and using three hash functions).
[More\...](structbght_1_1bcht.html#details)

`#include <bcht.hpp>`

+-----------------------------------+-----------------------------------+
| ## []{#pub-types} Public T        |                                   |
| ypes {#public-types .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **value_type** = pair\< Key, T \> |
| 4b3cd889e1796dc5d4968c384a3532b2} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Pair type.\                       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_type** = Key                |
| a6d9c3f77ff3f9684fe98b764a7fa257} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **mapped_type** = T               |
| 34bc9b1e3f7bccc1d6ab8ed2f94c52bf} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_type** =            |
| adc402d49fd3bae9f1efd23e97278190} | cuda::atomic\<                    |
| using                             | [value_type                       |
|                                   | ](structbght_1_1bcht.html#a4b3cd8 |
|                                   | 89e1796dc5d4968c384a3532b2){.el}, |
|                                   | Scope \>                          |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **allocator_type** = Allocator    |
| 0042b766b372f5d61e4474f679847c95} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **hasher** = Hash                 |
| bc80abbcc891e1b3127d247f353cdef4} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **size_type** = std::size_t       |
| 0f91a6131ebdf91b1b9013a3e39cfce1} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_allocator_type** =  |
| 4c03e045fb356f9814898e5fab277502} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | atomic_pair_type \>               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **pool_allocator_type** =         |
| 1614f4477cc58eb9d8468291a81b7b3f} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\< bool |
|                                   | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **size_type_allocator_type** =    |
| bda2b8b73ccf612e1686c631eafe8d9e} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | size_type \>                      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_equal** = KeyEqual          |
| 7f2e7dd47dc09db67be1c11962650bd6} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-method                 |                                   |
| s} Public Member Functions {#publ |                                   |
| ic-member-functions .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
|                                   | [bch                              |
|                                   | t](structbght_1_1bcht.html#ab77eb |
|                                   | f40b39673ff2943fbe950f6cee8){.el} |
|                                   | (std::size_t capacity, Key        |
|                                   | sentinel_key, T sentinel_value,   |
|                                   | Allocator const                   |
|                                   | &allocator=Allocator{})           |
+-----------------------------------+-----------------------------------+
|                                   | Constructs the hash table with    |
|                                   | the specified capacity and uses   |
|                                   | the specified sentinel key and    |
|                                   | value to define a sentinel pair.  |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#ab |
|                                   | 77ebf40b39673ff2943fbe950f6cee8)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **bcht** (const                   |
| ab4f5aaf7d1d1c16581c63425c22d98f} | [bc                               |
|                                   | ht](structbght_1_1bcht.html){.el} |
|                                   | &other)                           |
+-----------------------------------+-----------------------------------+
|                                   | A shallow-copy constructor.\      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **bcht**                          |
| fdc822e8c97860d7d63b30ff66cd44a7} | ([bc                              |
|                                   | ht](structbght_1_1bcht.html){.el} |
|                                   | &&)=delete                        |
+-----------------------------------+-----------------------------------+
|                                   | Move constructor is currently     |
|                                   | deleted.\                         |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=** (const              |
| 2b193d98eb00b0c41900ff06cbb80fde} | [bc                               |
| [bc                               | ht](structbght_1_1bcht.html){.el} |
| ht](structbght_1_1bcht.html){.el} | &)=delete                         |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The assignment operator for the   |
|                                   | BCHT is currently deleted.\       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=**                     |
| b91884025ae177f74ad6938789c23f3a} | ([bc                              |
| [bc                               | ht](structbght_1_1bcht.html){.el} |
| ht](structbght_1_1bcht.html){.el} | &&)=delete                        |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The move assignment operator for  |
|                                   | the BCHT is currently deleted.\   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **\~bcht** ()                     |
| 99a897955e80d66cb83d62ef33d79110} |                                   |
|                                   |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Destructor that destroys the hash |
|                                   | map and deallocate memory if no   |
|                                   | copies exist.\                    |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **clear** ()                      |
| 798fc4be13d418089ac8d491e98b5147} |                                   |
| void                              |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Clears the hash map and resets    |
|                                   | all slots.\                       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt \>     |                                   |
+-----------------------------------+-----------------------------------+
| bool                              | [inser                            |
|                                   | t](structbght_1_1bcht.html#a2e8e2 |
|                                   | 1183a7978f2908a801f98138d22){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for inserting all   |
|                                   | pairs defined by the input        |
|                                   | argument iterators. All keys in   |
|                                   | the range must be unique and must |
|                                   | not exist in the hash table.      |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#a2 |
|                                   | e8e21183a7978f2908a801f98138d22)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt ,      |                                   |
| typename OutputIt \>              |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [fin                              |
|                                   | d](structbght_1_1bcht.html#aa00dd |
|                                   | 13fd68144e59cfdd1bd0ce212dc){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | OutputIt output_begin,            |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for finding all     |
|                                   | keys defined by the input         |
|                                   | argument iterators.               |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#aa |
|                                   | 00dd13fd68144e59cfdd1bd0ce212dc)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename tile_type \>   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ bool               | [inser                            |
|                                   | t](structbght_1_1bcht.html#a8ca06 |
|                                   | cef309ce158a21ba686088a5804){.el} |
|                                   | ([value_typ                       |
|                                   | e](structbght_1_1bcht.html#a4b3cd |
|                                   | 889e1796dc5d4968c384a3532b2){.el} |
|                                   | const &pair, tile_type const      |
|                                   | &tile)                            |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative insertion |
|                                   | API that inserts a single pair    |
|                                   | into the hash map.                |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#a8 |
|                                   | ca06cef309ce158a21ba686088a5804)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename tile_type \>   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ mapped_type        | [fin                              |
|                                   | d](structbght_1_1bcht.html#adbbad |
|                                   | f9704bd5752b55fe0e7ab9d788c){.el} |
|                                   | (key_type const &key, tile_type   |
|                                   | const &tile)                      |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative find API  |
|                                   | that finds a single pair into the |
|                                   | hash map.                         |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#ad |
|                                   | bbadf9704bd5752b55fe0e7ab9d788c)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename RNG \>         |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [randomize_hash_function          |
|                                   | s](structbght_1_1bcht.html#afceb3 |
|                                   | 1cd69d17b0051edd59550babd23){.el} |
|                                   | (RNG &rng)                        |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API to randomize the    |
|                                   | hash functions used for the       |
|                                   | probing scheme. This can be used  |
|                                   | when the hash table construction  |
|                                   | fails. The hash table must be     |
|                                   | cleared after a call to this      |
|                                   | function.                         |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#af |
|                                   | ceb31cd69d17b0051edd59550babd23)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| size_type                         | [siz                              |
|                                   | e](structbght_1_1bcht.html#a4fc21 |
|                                   | 878958c178a66d686039c18f957){.el} |
|                                   | (cudaStream_t stream=0)           |
+-----------------------------------+-----------------------------------+
|                                   | Compute the number of elements in |
|                                   | the map.                          |
|                                   | [Mor                              |
|                                   | e\...](structbght_1_1bcht.html#a4 |
|                                   | fc21878958c178a66d686039c18f957)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-static-attribs}        |                                   |
|  Static Public Attributes {#stati |                                   |
| c-public-attributes .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **bucket_size** = B               |
| dab32be4f06400b48898b2ee2e0bc54a} |                                   |
| static constexpr auto             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#friends                    |                                   |
| } Friends {#friends .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             |                                   |
| 61023bda7074c9fd00d6e249fd9b4f3d} |                                   |
| template\<typename InputIt ,      |                                   |
| typename HashMap \>               |                                   |
+-----------------------------------+-----------------------------------+
| \_\_global\_\_ void               | **detai                           |
|                                   | l::kernels::tiled_insert_kernel** |
|                                   | (InputIt, InputIt, HashMap)       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             |                                   |
| 747d3ab2974582ff7a796c479ad7c51f} |                                   |
| template\<typename InputIt ,      |                                   |
| typename OutputIt , typename      |                                   |
| HashMap \>                        |                                   |
+-----------------------------------+-----------------------------------+
| \_\_global\_\_ void               | **det                             |
|                                   | ail::kernels::tiled_find_kernel** |
|                                   | (InputIt, InputIt, OutputIt,      |
|                                   | HashMap)                          |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             |                                   |
| 00965174f563177b83a451032185a46a} |                                   |
| template\<int BlockSize, typename |                                   |
| InputT , typename HashMap \>      |                                   |
+-----------------------------------+-----------------------------------+
| \_\_global\_\_ void               | **detail::kernels::count_kernel** |
|                                   | (const InputT, std::size_t \*,    |
|                                   | HashMap)                          |
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
struct bght::bcht\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>
:::

BCHT BCHT (bucketed cuckoo hash table) is an associative static GPU hash
table that contains key-value pairs with unique keys. The hash table is
an open addressing hash table based on the cuckoo hashing probing scheme
(bucketed and using three hash functions).

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

[]{#ab77ebf40b39673ff2943fbe950f6cee8}

## [[◆ ](#ab77ebf40b39673ff2943fbe950f6cee8)]{.permalink}bcht() {#bcht .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

  ----------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ------------------------------
  [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::[bcht](structbght_1_1bcht.html){.el}   (   std::size_t          *capacity*,
                                                                                                                                          Key                  *sentinel_key*,
                                                                                                                                          T                    *sentinel_value*,
                                                                                                                                          Allocator const &    *allocator* = `Allocator{}` 
                                                                                                                                      )                        
  ----------------------------------------------------------------------------------------------------------------------------------- --- -------------------- ------------------------------
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

[]{#aa00dd13fd68144e59cfdd1bd0ce212dc}

## [[◆ ](#aa00dd13fd68144e59cfdd1bd0ce212dc)]{.permalink}find() [\[1/2\]]{.overload} {#find-12 .memtitle}

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

  -------------------------------------------------------------------------------------------------------- --- --------------- -----------------
  void [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::find   (   InputIt         *first*,
                                                                                                               InputIt         *last*,
                                                                                                               OutputIt        *output_begin*,
                                                                                                               cudaStream_t    *stream* = `0` 
                                                                                                           )                   
  -------------------------------------------------------------------------------------------------------- --- --------------- -----------------
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

[]{#adbbadf9704bd5752b55fe0e7ab9d788c}

## [[◆ ](#adbbadf9704bd5752b55fe0e7ab9d788c)]{.permalink}find() [\[2/2\]]{.overload} {#find-22 .memtitle}

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

  ------------------------------------------------------------------------------------------------------------------------------ --- -------------------- ---------
  \_\_device\_\_ mapped_type [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::find   (   key_type const &     *key*,
                                                                                                                                     tile_type const &    *tile* 
                                                                                                                                 )                        
  ------------------------------------------------------------------------------------------------------------------------------ --- -------------------- ---------
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

[]{#a2e8e21183a7978f2908a801f98138d22}

## [[◆ ](#a2e8e21183a7978f2908a801f98138d22)]{.permalink}insert() [\[1/2\]]{.overload} {#insert-12 .memtitle}

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

  ---------------------------------------------------------------------------------------------------------- --- --------------- -----------------
  bool [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::insert   (   InputIt         *first*,
                                                                                                                 InputIt         *last*,
                                                                                                                 cudaStream_t    *stream* = `0` 
                                                                                                             )                   
  ---------------------------------------------------------------------------------------------------------- --- --------------- -----------------
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

[]{#a8ca06cef309ce158a21ba686088a5804}

## [[◆ ](#a8ca06cef309ce158a21ba686088a5804)]{.permalink}insert() [\[2/2\]]{.overload} {#insert-22 .memtitle}

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

  ------------------------------------------------------------------------------------------------------------------------- --- --------------------------------------------------------------------------------------- ---------
  \_\_device\_\_ bool [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::insert   (   [value_type](structbght_1_1bcht.html#a4b3cd889e1796dc5d4968c384a3532b2){.el} const &    *pair*,
                                                                                                                                tile_type const &                                                                       *tile* 
                                                                                                                            )                                                                                           
  ------------------------------------------------------------------------------------------------------------------------- --- --------------------------------------------------------------------------------------- ---------
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

[]{#afceb31cd69d17b0051edd59550babd23}

## [[◆ ](#afceb31cd69d17b0051edd59550babd23)]{.permalink}randomize_hash_functions() {#randomize_hash_functions .memtitle}

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

  ---------------------------------------------------------------------------------------------------------------------------- --- -------- ------- --- --
  void [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::randomize_hash_functions   (   RNG &    *rng*   )   
  ---------------------------------------------------------------------------------------------------------------------------- --- -------- ------- --- --
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

[]{#a4fc21878958c178a66d686039c18f957}

## [[◆ ](#a4fc21878958c178a66d686039c18f957)]{.permalink}size() {#size .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>, int B = 16\>
:::

  ------------------------------------------------------------------------------------------------------------- --- --------------- ---------------- --- --
  size_type [bght::bcht](structbght_1_1bcht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator, B \>::size   (   cudaStream_t    *stream* = `0`   )   
  ------------------------------------------------------------------------------------------------------------- --- --------------- ---------------- --- --
:::

::: memdoc
Compute the number of elements in the map.

Returns
:   The number of elements in the map
:::
:::

------------------------------------------------------------------------

The documentation for this struct was generated from the following file:

-   include/[bcht.hpp](bcht_8hpp_source.html){.el}
:::

------------------------------------------------------------------------

[Generated by [![doxygen](doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
