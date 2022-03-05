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
-   [cht](../../d3/dad/structbght_1_1cht.html){.el}
:::
:::

::: header
::: summary
[Public Types](#pub-types) \| [Public Member Functions](#pub-methods)
:::

::: headertitle
::: title
bght::cht\< Key, T, Hash, KeyEqual, Scope, Allocator \> Struct Template
Reference
:::
:::
:::

::: contents
CHT CHT (cuckoo hash table) is an associative static GPU hash table that
contains key-value pairs with unique keys. The hash table is an open
addressing hash table based on the cuckoo hashing probing scheme (bucket
size of one and using four hash functions).
[More\...](../../d3/dad/structbght_1_1cht.html#details)

`#include <cht.hpp>`

+-----------------------------------+-----------------------------------+
| ## []{#pub-types} Public T        |                                   |
| ypes {#public-types .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **value_type** = pair\< Key, T \> |
| f96deb884c2a139122cb79fc1f4899e1} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_type** = Key                |
| 1f25a7b0b7e8f73ae30cd788f6ceeb54} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **mapped_type** = T               |
| 3d59f72cee1eda4593e772a12984701c} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_type** =            |
| abea41ba24720eecf390550a8f684faf} | cuda::atomic\< value_type, Scope  |
| using                             | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **allocator_type** = Allocator    |
| 66762f6f791165c92d74757ce1d35122} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **hasher** = Hash                 |
| 6e730e9035c26900f69916beb9ef5db0} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **atomic_pair_allocator_type** =  |
| 510541b04a915afd551d2c9213388dd6} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\<      |
|                                   | atomic_pair_type \>               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **pool_allocator_type** =         |
| 5191a78aea4516b2582c99e9079eb5f9} | typename std::allocator_traits\<  |
| using                             | Allocator \>::rebind_alloc\< bool |
|                                   | \>                                |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **key_equal** = KeyEqual          |
| 80fa4acebf24ed3f1f923afe54d7f155} |                                   |
| using                             |                                   |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+

+-----------------------------------+-----------------------------------+
| ## []{#pub-method                 |                                   |
| s} Public Member Functions {#publ |                                   |
| ic-member-functions .groupheader} |                                   |
+-----------------------------------+-----------------------------------+
|                                   | [cht](../../d3/                   |
|                                   | dad/structbght_1_1cht.html#aeba94 |
|                                   | f76492153267d6c2a8f0de331b5){.el} |
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
|                                   | /d3/dad/structbght_1_1cht.html#ae |
|                                   | ba94f76492153267d6c2a8f0de331b5)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **cht** (const                    |
| b8475ae4c11257ff5476e6a5329a7af8} | [cht](../../d3                    |
|                                   | /dad/structbght_1_1cht.html){.el} |
|                                   | &other)                           |
+-----------------------------------+-----------------------------------+
|                                   | A shallow-copy constructor.\      |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **cht**                           |
| 87299241de2004dfe0bc7662401d5ea1} | ([cht](../../d3                   |
|                                   | /dad/structbght_1_1cht.html){.el} |
|                                   | &&)=delete                        |
+-----------------------------------+-----------------------------------+
|                                   | Move constructor is currently     |
|                                   | deleted.\                         |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=** (const              |
| 090f8a449a06b62f43057d7a3fddea99} | [cht](../../d3                    |
| [cht](../../d3                    | /dad/structbght_1_1cht.html){.el} |
| /dad/structbght_1_1cht.html){.el} | &)=delete                         |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The assignment operator is        |
|                                   | currently deleted.\               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **operator=**                     |
| d62008d5e9b37455c84620b908b88bdf} | ([cht](../../d3                   |
| [cht](../../d3                    | /dad/structbght_1_1cht.html){.el} |
| /dad/structbght_1_1cht.html){.el} | &&)=delete                        |
| &                                 |                                   |
+-----------------------------------+-----------------------------------+
|                                   | The move assignment operator is   |
|                                   | currently deleted.\               |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **\~cht** ()                      |
| 4e5e1879dbce693b73fd9569e9d86a34} |                                   |
|                                   |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Destructor that destroys the hash |
|                                   | map and deallocate memory if no   |
|                                   | copies exist.\                    |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| []{#a                             | **clear** ()                      |
| 28b618c3f217d021318f097dab537c4a} |                                   |
| void                              |                                   |
+-----------------------------------+-----------------------------------+
|                                   | Clears the hash map and resets    |
|                                   | all slots.\                       |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt \>     |                                   |
+-----------------------------------+-----------------------------------+
| bool                              | [insert](../../d3/                |
|                                   | dad/structbght_1_1cht.html#a33b47 |
|                                   | bb3d9abe89ab834e04b210f571c){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for inserting all   |
|                                   | pairs defined by the input        |
|                                   | argument iterators. All keys in   |
|                                   | the range must be unique and must |
|                                   | not exist in the hash table.      |
|                                   | [More\...](../..                  |
|                                   | /d3/dad/structbght_1_1cht.html#a3 |
|                                   | 3b47bb3d9abe89ab834e04b210f571c)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename InputIt ,      |                                   |
| typename OutputIt \>              |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [find](../../d3/                  |
|                                   | dad/structbght_1_1cht.html#a81fdd |
|                                   | a2b20c3dc2fd2c237e1e8c9ccac){.el} |
|                                   | (InputIt first, InputIt last,     |
|                                   | OutputIt output_begin,            |
|                                   | cudaStream_t stream=0)            |
+-----------------------------------+-----------------------------------+
|                                   | Host-side API for finding all     |
|                                   | keys defined by the input         |
|                                   | argument iterators.               |
|                                   | [More\...](../..                  |
|                                   | /d3/dad/structbght_1_1cht.html#a8 |
|                                   | 1fdda2b20c3dc2fd2c237e1e8c9ccac)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ bool               | [insert](../../d3/                |
|                                   | dad/structbght_1_1cht.html#a8a868 |
|                                   | 596bd28694f9885d410ef3b37a3){.el} |
|                                   | (value_type const &pair)          |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative insertion |
|                                   | API that inserts a single pair    |
|                                   | into the hash map. This function  |
|                                   | is called by a single thread.     |
|                                   | [More\...](../..                  |
|                                   | /d3/dad/structbght_1_1cht.html#a8 |
|                                   | a868596bd28694f9885d410ef3b37a3)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| \_\_device\_\_ mapped_type        | [find](../../d3/                  |
|                                   | dad/structbght_1_1cht.html#a6f45a |
|                                   | 140a4f942230ffb4609096982ec){.el} |
|                                   | (key_type const &key)             |
+-----------------------------------+-----------------------------------+
|                                   | Device-side cooperative find API  |
|                                   | that finds a single pair into the |
|                                   | hash map.                         |
|                                   | [More\...](../..                  |
|                                   | /d3/dad/structbght_1_1cht.html#a6 |
|                                   | f45a140a4f942230ffb4609096982ec)\ |
+-----------------------------------+-----------------------------------+
|                                   |                                   |
+-----------------------------------+-----------------------------------+
| template\<typename RNG \>         |                                   |
+-----------------------------------+-----------------------------------+
| void                              | [ra                               |
|                                   | ndomize_hash_functions](../../d3/ |
|                                   | dad/structbght_1_1cht.html#a9e7ca |
|                                   | 5b7d83f47e4188ffbeacbc4b7dc){.el} |
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
|                                   | /d3/dad/structbght_1_1cht.html#a9 |
|                                   | e7ca5b7d83f47e4188ffbeacbc4b7dc)\ |
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
bght::cuda_allocator\<char\>\>\
struct bght::cht\< Key, T, Hash, KeyEqual, Scope, Allocator \>
:::

CHT CHT (cuckoo hash table) is an associative static GPU hash table that
contains key-value pairs with unique keys. The hash table is an open
addressing hash table based on the cuckoo hashing probing scheme (bucket
size of one and using four hash functions).

Template Parameters

:   ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Key         Type for the hash map key
      T           Type for the mapped value
      Hash        Unary function object class that defines the hash function. The function must have an `initialize_hf` specialization to initialize the hash function using a random number generator
      KeyEqual    Binary function object class that compares two keys
      Allocator   The allocator to use for allocating GPU device memory
      ----------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

## Constructor & Destructor Documentation {#constructor-destructor-documentation .groupheader}

[]{#aeba94f76492153267d6c2a8f0de331b5}

## [[◆ ](#aeba94f76492153267d6c2a8f0de331b5)]{.permalink}cht() {#cht .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>\>
:::

  ------------------------------------------------------------------------------------------------------------------------------------------------------ --- -------------------- ------------------------------
  [bght::cht](../../d3/dad/structbght_1_1cht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator \>::[cht](../../d3/dad/structbght_1_1cht.html){.el}   (   std::size_t          *capacity*,
                                                                                                                                                             Key                  *sentinel_key*,
                                                                                                                                                             T                    *sentinel_value*,
                                                                                                                                                             Allocator const &    *allocator* = `Allocator{}` 
                                                                                                                                                         )                        
  ------------------------------------------------------------------------------------------------------------------------------------------------------ --- -------------------- ------------------------------
:::

::: memdoc
Constructs the hash table with the specified capacity and uses the
specified sentinel key and value to define a sentinel pair.

Parameters

:   ---------------- -------------------------------------------------------
      capacity         The number of slots to use in the hash table
      sentinel_key     A reserved sentinel key that defines an empty key
      sentinel_value   A reserved sentinel value that defines an empty value
      allocator        The allocator to use for allocating GPU device memory
      ---------------- -------------------------------------------------------
:::
:::

## Member Function Documentation {#member-function-documentation .groupheader}

[]{#a81fdda2b20c3dc2fd2c237e1e8c9ccac}

## [[◆ ](#a81fdda2b20c3dc2fd2c237e1e8c9ccac)]{.permalink}find() [\[1/2\]]{.overload} {#find-12 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>\>
:::

::: memtemplate
template\<typename InputIt , typename OutputIt \>
:::

  ---------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
  void [bght::cht](../../d3/dad/structbght_1_1cht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator \>::find   (   InputIt         *first*,
                                                                                                                       InputIt         *last*,
                                                                                                                       OutputIt        *output_begin*,
                                                                                                                       cudaStream_t    *stream* = `0` 
                                                                                                                   )                   
  ---------------------------------------------------------------------------------------------------------------- --- --------------- -----------------
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

[]{#a6f45a140a4f942230ffb4609096982ec}

## [[◆ ](#a6f45a140a4f942230ffb4609096982ec)]{.permalink}find() [\[2/2\]]{.overload} {#find-22 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>\>
:::

  -------------------------------------------------------------------------------------------------------------------------------------- --- ------------------- ------- --- --
  \_\_device\_\_ mapped_type [bght::cht](../../d3/dad/structbght_1_1cht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator \>::find   (   key_type const &    *key*   )   
  -------------------------------------------------------------------------------------------------------------------------------------- --- ------------------- ------- --- --
:::

::: memdoc
Device-side cooperative find API that finds a single pair into the hash
map.

Parameters

:   ----- -------------------------------------------------------------------------------------------------------
      key   A key to find in the hash map. The key must be the same for all threads in the cooperative group tile
      ----- -------------------------------------------------------------------------------------------------------

```{=html}
<!-- -->
```

Returns
:   The value of the key if it exists in the map or the `sentinel_value`
    if the key does not exist in the hash map
:::
:::

[]{#a33b47bb3d9abe89ab834e04b210f571c}

## [[◆ ](#a33b47bb3d9abe89ab834e04b210f571c)]{.permalink}insert() [\[1/2\]]{.overload} {#insert-12 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>\>
:::

::: memtemplate
template\<typename InputIt \>
:::

  ------------------------------------------------------------------------------------------------------------------ --- --------------- -----------------
  bool [bght::cht](../../d3/dad/structbght_1_1cht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator \>::insert   (   InputIt         *first*,
                                                                                                                         InputIt         *last*,
                                                                                                                         cudaStream_t    *stream* = `0` 
                                                                                                                     )                   
  ------------------------------------------------------------------------------------------------------------------ --- --------------- -----------------
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

[]{#a8a868596bd28694f9885d410ef3b37a3}

## [[◆ ](#a8a868596bd28694f9885d410ef3b37a3)]{.permalink}insert() [\[2/2\]]{.overload} {#insert-22 .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>\>
:::

  --------------------------------------------------------------------------------------------------------------------------------- --- --------------------- -------- --- --
  \_\_device\_\_ bool [bght::cht](../../d3/dad/structbght_1_1cht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator \>::insert   (   value_type const &    *pair*   )   
  --------------------------------------------------------------------------------------------------------------------------------- --- --------------------- -------- --- --
:::

::: memdoc
Device-side cooperative insertion API that inserts a single pair into
the hash map. This function is called by a single thread.

Parameters

:   ------ -----------------------------------------------
      pair   A key-value pair to insert into the hash map.
      ------ -----------------------------------------------

```{=html}
<!-- -->
```

Returns
:   A boolean indicating success (true) or failure (false) of the
    insertion operation.
:::
:::

[]{#a9e7ca5b7d83f47e4188ffbeacbc4b7dc}

## [[◆ ](#a9e7ca5b7d83f47e4188ffbeacbc4b7dc)]{.permalink}randomize_hash_functions() {#randomize_hash_functions .memtitle}

::: memitem
::: memproto
::: memtemplate
template\<class Key , class T , class Hash =
bght::universal_hash\<Key\>, class KeyEqual = bght::equal_to\<Key\>,
cuda::thread_scope Scope = cuda::thread_scope_device, class Allocator =
bght::cuda_allocator\<char\>\>
:::

::: memtemplate
template\<typename RNG \>
:::

  ------------------------------------------------------------------------------------------------------------------------------------ --- -------- ------- --- --
  void [bght::cht](../../d3/dad/structbght_1_1cht.html){.el}\< Key, T, Hash, KeyEqual, Scope, Allocator \>::randomize_hash_functions   (   RNG &    *rng*   )   
  ------------------------------------------------------------------------------------------------------------------------------------ --- -------- ------- --- --
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

------------------------------------------------------------------------

The documentation for this struct was generated from the following file:

-   include/[cht.hpp](../../d3/dd0/cht_8hpp_source.html){.el}
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
