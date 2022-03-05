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
-   [include](dir_d44c64559bbebec7f509842c48db8b23.html){.el}
:::
:::

::: header
::: headertitle
::: title
cht.hpp
:::
:::
:::

::: contents
::: fragment
::: line
[]{#l00001}[ 1]{.lineno}[/\*]{.comment}
:::

::: line
[]{#l00002}[ 2]{.lineno}[ \* Copyright 2021 The Regents of the
University of California, Davis]{.comment}
:::

::: line
[]{#l00003}[ 3]{.lineno}[ \*]{.comment}
:::

::: line
[]{#l00004}[ 4]{.lineno}[ \* Licensed under the Apache License, Version
2.0 (the \"License\");]{.comment}
:::

::: line
[]{#l00005}[ 5]{.lineno}[ \* you may not use this file except in
compliance with the License.]{.comment}
:::

::: line
[]{#l00006}[ 6]{.lineno}[ \* You may obtain a copy of the License
at]{.comment}
:::

::: line
[]{#l00007}[ 7]{.lineno}[ \*]{.comment}
:::

::: line
[]{#l00008}[ 8]{.lineno}[ \*
http://www.apache.org/licenses/LICENSE-2.0]{.comment}
:::

::: line
[]{#l00009}[ 9]{.lineno}[ \*]{.comment}
:::

::: line
[]{#l00010}[ 10]{.lineno}[ \* Unless required by applicable law or
agreed to in writing, software]{.comment}
:::

::: line
[]{#l00011}[ 11]{.lineno}[ \* distributed under the License is
distributed on an \"AS IS\" BASIS,]{.comment}
:::

::: line
[]{#l00012}[ 12]{.lineno}[ \* WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.]{.comment}
:::

::: line
[]{#l00013}[ 13]{.lineno}[ \* See the License for the specific language
governing permissions and]{.comment}
:::

::: line
[]{#l00014}[ 14]{.lineno}[ \* limitations under the License.]{.comment}
:::

::: line
[]{#l00015}[ 15]{.lineno}[ \*/]{.comment}
:::

::: line
[]{#l00016}[ 16]{.lineno}
:::

::: line
[]{#l00017}[ 17]{.lineno}[#pragma once]{.preprocessor}
:::

::: line
[]{#l00018}[ 18]{.lineno}[#include \<cuda/atomic\>]{.preprocessor}
:::

::: line
[]{#l00019}[ 19]{.lineno}[#include \<cuda/std/utility\>]{.preprocessor}
:::

::: line
[]{#l00020}[ 20]{.lineno}[#include
\<detail/allocator.hpp\>]{.preprocessor}
:::

::: line
[]{#l00021}[ 21]{.lineno}[#include
\<detail/cuda_helpers.cuh\>]{.preprocessor}
:::

::: line
[]{#l00022}[ 22]{.lineno}[#include
\<detail/hash_functions.cuh\>]{.preprocessor}
:::

::: line
[]{#l00023}[ 23]{.lineno}[#include
\<detail/kernels.cuh\>]{.preprocessor}
:::

::: line
[]{#l00024}[ 24]{.lineno}[#include \<detail/pair.cuh\>]{.preprocessor}
:::

::: line
[]{#l00025}[ 25]{.lineno}[#include \<memory\>]{.preprocessor}
:::

::: line
[]{#l00026}[ 26]{.lineno}
:::

::: line
[]{#l00027}[ 27]{.lineno}[namespace ]{.keyword}bght {
:::

::: line
[]{#l00028}[ 28]{.lineno}
:::

::: line
[]{#l00043}[ 43]{.lineno}[template]{.keyword} \<[class ]{.keyword}Key,
:::

::: line
[]{#l00044}[ 44]{.lineno} [class ]{.keyword}T,
:::

::: line
[]{#l00045}[ 45]{.lineno} [class ]{.keyword}Hash =
bght::universal_hash\<Key\>,
:::

::: line
[]{#l00046}[ 46]{.lineno} [class ]{.keyword}KeyEqual =
bght::equal_to\<Key\>,
:::

::: line
[]{#l00047}[ 47]{.lineno} cuda::thread_scope Scope =
cuda::thread_scope_device,
:::

::: line
[]{#l00048}[ 48]{.lineno} [class ]{.keyword}Allocator =
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class}\>
:::

::: line
[]{#l00049}[ [49](structbght_1_1cht.html){.line}]{.lineno}[struct
]{.keyword}[cht](structbght_1_1cht.html){.code .hl_struct} {
:::

::: line
[]{#l00050}[ 50]{.lineno} [using]{.keyword} value_type = pair\<Key, T\>;
:::

::: line
[]{#l00051}[ 51]{.lineno} [using]{.keyword} key_type = Key;
:::

::: line
[]{#l00052}[ 52]{.lineno} [using]{.keyword} mapped_type = T;
:::

::: line
[]{#l00053}[ 53]{.lineno} [using]{.keyword} atomic_pair_type =
cuda::atomic\<value_type, Scope\>;
:::

::: line
[]{#l00054}[ 54]{.lineno} [using]{.keyword} allocator_type = Allocator;
:::

::: line
[]{#l00055}[ 55]{.lineno} [using]{.keyword} hasher = Hash;
:::

::: line
[]{#l00056}[ 56]{.lineno} [using]{.keyword} atomic_pair_allocator_type =
:::

::: line
[]{#l00057}[ 57]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<atomic_pair_type\>;
:::

::: line
[]{#l00058}[ 58]{.lineno} [using]{.keyword} pool_allocator_type =
:::

::: line
[]{#l00059}[ 59]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<[bool]{.keywordtype}\>;
:::

::: line
[]{#l00060}[ 60]{.lineno} [using]{.keyword} key_equal = KeyEqual;
:::

::: line
[]{#l00061}[ 61]{.lineno}
:::

::: line
[]{#l00071}[
[71](structbght_1_1cht.html#aeba94f76492153267d6c2a8f0de331b5){.line}]{.lineno}
[cht](structbght_1_1cht.html#aeba94f76492153267d6c2a8f0de331b5){.code
.hl_function}(std::size_t capacity,
:::

::: line
[]{#l00072}[ 72]{.lineno} Key sentinel_key,
:::

::: line
[]{#l00073}[ 73]{.lineno} T sentinel_value,
:::

::: line
[]{#l00074}[ 74]{.lineno} Allocator [const]{.keyword}& allocator =
Allocator{});
:::

::: line
[]{#l00075}[ 75]{.lineno}
:::

::: line
[]{#l00079}[
[79](structbght_1_1cht.html#ab8475ae4c11257ff5476e6a5329a7af8){.line}]{.lineno}
[cht](structbght_1_1cht.html#ab8475ae4c11257ff5476e6a5329a7af8){.code
.hl_function}([const]{.keyword} [cht](structbght_1_1cht.html){.code
.hl_struct}& other);
:::

::: line
[]{#l00083}[
[83](structbght_1_1cht.html#a87299241de2004dfe0bc7662401d5ea1){.line}]{.lineno}
[cht](structbght_1_1cht.html#a87299241de2004dfe0bc7662401d5ea1){.code
.hl_function}([cht](structbght_1_1cht.html){.code .hl_struct}&&) =
[delete]{.keyword};
:::

::: line
[]{#l00087}[
[87](structbght_1_1cht.html#a090f8a449a06b62f43057d7a3fddea99){.line}]{.lineno}
[cht](structbght_1_1cht.html){.code .hl_struct}&
[operator=](structbght_1_1cht.html#a090f8a449a06b62f43057d7a3fddea99){.code
.hl_function}([const]{.keyword} [cht](structbght_1_1cht.html){.code
.hl_struct}&) = [delete]{.keyword};
:::

::: line
[]{#l00091}[
[91](structbght_1_1cht.html#ad62008d5e9b37455c84620b908b88bdf){.line}]{.lineno}
[cht](structbght_1_1cht.html){.code .hl_struct}&
[operator=](structbght_1_1cht.html#ad62008d5e9b37455c84620b908b88bdf){.code
.hl_function}([cht](structbght_1_1cht.html){.code .hl_struct}&&) =
[delete]{.keyword};
:::

::: line
[]{#l00095}[
[95](structbght_1_1cht.html#a4e5e1879dbce693b73fd9569e9d86a34){.line}]{.lineno}
[\~cht](structbght_1_1cht.html#a4e5e1879dbce693b73fd9569e9d86a34){.code
.hl_function}();
:::

::: line
[]{#l00099}[
[99](structbght_1_1cht.html#a28b618c3f217d021318f097dab537c4a){.line}]{.lineno}
[void]{.keywordtype}
[clear](structbght_1_1cht.html#a28b618c3f217d021318f097dab537c4a){.code
.hl_function}();
:::

::: line
[]{#l00100}[ 100]{.lineno}
:::

::: line
[]{#l00111}[ 111]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt\>
:::

::: line
[]{#l00112}[
[112](structbght_1_1cht.html#a33b47bb3d9abe89ab834e04b210f571c){.line}]{.lineno}
[bool]{.keywordtype}
[insert](structbght_1_1cht.html#a33b47bb3d9abe89ab834e04b210f571c){.code
.hl_function}(InputIt first, InputIt last, cudaStream_t stream = 0);
:::

::: line
[]{#l00113}[ 113]{.lineno}
:::

::: line
[]{#l00125}[ 125]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt\>
:::

::: line
[]{#l00126}[
[126](structbght_1_1cht.html#a81fdda2b20c3dc2fd2c237e1e8c9ccac){.line}]{.lineno}
[void]{.keywordtype}
[find](structbght_1_1cht.html#a81fdda2b20c3dc2fd2c237e1e8c9ccac){.code
.hl_function}(InputIt first, InputIt last, OutputIt output_begin,
cudaStream_t stream = 0);
:::

::: line
[]{#l00127}[ 127]{.lineno}
:::

::: line
[]{#l00135}[
[135](structbght_1_1cht.html#a8a868596bd28694f9885d410ef3b37a3){.line}]{.lineno}
\_\_device\_\_ [bool]{.keywordtype}
[insert](structbght_1_1cht.html#a8a868596bd28694f9885d410ef3b37a3){.code
.hl_function}(value_type [const]{.keyword}& pair);
:::

::: line
[]{#l00136}[ 136]{.lineno}
:::

::: line
[]{#l00145}[
[145](structbght_1_1cht.html#a6f45a140a4f942230ffb4609096982ec){.line}]{.lineno}
\_\_device\_\_ mapped_type
[find](structbght_1_1cht.html#a6f45a140a4f942230ffb4609096982ec){.code
.hl_function}(key_type [const]{.keyword}& key);
:::

::: line
[]{#l00146}[ 146]{.lineno}
:::

::: line
[]{#l00154}[ 154]{.lineno} [template]{.keyword} \<[typename]{.keyword}
RNG\>
:::

::: line
[]{#l00155}[
[155](structbght_1_1cht.html#a9e7ca5b7d83f47e4188ffbeacbc4b7dc){.line}]{.lineno}
[void]{.keywordtype}
[randomize_hash_functions](structbght_1_1cht.html#a9e7ca5b7d83f47e4188ffbeacbc4b7dc){.code
.hl_function}(RNG& rng);
:::

::: line
[]{#l00156}[ 156]{.lineno}
:::

::: line
[]{#l00157}[ 157]{.lineno} [private]{.keyword}:
:::

::: line
[]{#l00158}[ 158]{.lineno} \_\_device\_\_ [void]{.keywordtype}
set_build_success([const]{.keyword} [bool]{.keywordtype}& success) {
\*d_build_success\_ = success; }
:::

::: line
[]{#l00159}[ 159]{.lineno}
:::

::: line
[]{#l00160}[ 160]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00161}[ 161]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::insert_kernel(InputIt, InputIt,
HashMap);
:::

::: line
[]{#l00162}[ 162]{.lineno}
:::

::: line
[]{#l00163}[ 163]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00164}[ 164]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::find_kernel(InputIt,
:::

::: line
[]{#l00165}[ 165]{.lineno} InputIt,
:::

::: line
[]{#l00166}[ 166]{.lineno} OutputIt,
:::

::: line
[]{#l00167}[ 167]{.lineno} HashMap);
:::

::: line
[]{#l00168}[ 168]{.lineno}
:::

::: line
[]{#l00169}[ 169]{.lineno} std::size_t capacity\_;
:::

::: line
[]{#l00170}[ 170]{.lineno} key_type sentinel_key\_{};
:::

::: line
[]{#l00171}[ 171]{.lineno} mapped_type sentinel_value\_{};
:::

::: line
[]{#l00172}[ 172]{.lineno} allocator_type allocator\_;
:::

::: line
[]{#l00173}[ 173]{.lineno} atomic_pair_allocator_type
atomic_pairs_allocator\_;
:::

::: line
[]{#l00174}[ 174]{.lineno} pool_allocator_type pool_allocator\_;
:::

::: line
[]{#l00175}[ 175]{.lineno}
:::

::: line
[]{#l00176}[ 176]{.lineno} atomic_pair_type\* d_table\_{};
:::

::: line
[]{#l00177}[ 177]{.lineno} std::shared_ptr\<atomic_pair_type\> table\_;
:::

::: line
[]{#l00178}[ 178]{.lineno}
:::

::: line
[]{#l00179}[ 179]{.lineno} [bool]{.keywordtype}\* d_build_success\_;
:::

::: line
[]{#l00180}[ 180]{.lineno} std::shared_ptr\<bool\> build_success\_;
:::

::: line
[]{#l00181}[ 181]{.lineno}
:::

::: line
[]{#l00182}[ 182]{.lineno} uint32_t max_cuckoo_chains\_;
:::

::: line
[]{#l00183}[ 183]{.lineno}
:::

::: line
[]{#l00184}[ 184]{.lineno} Hash hf0\_;
:::

::: line
[]{#l00185}[ 185]{.lineno} Hash hf1\_;
:::

::: line
[]{#l00186}[ 186]{.lineno} Hash hf2\_;
:::

::: line
[]{#l00187}[ 187]{.lineno} Hash hf3\_;
:::

::: line
[]{#l00188}[ 188]{.lineno}
:::

::: line
[]{#l00189}[ 189]{.lineno} std::size_t num_buckets\_;
:::

::: line
[]{#l00190}[ 190]{.lineno}};
:::

::: line
[]{#l00191}[ 191]{.lineno}
:::

::: line
[]{#l00192}[ 192]{.lineno}} [// namespace bght]{.comment}
:::

::: line
[]{#l00193}[ 193]{.lineno}
:::

::: line
[]{#l00194}[ 194]{.lineno}[#include
\<detail/cht_impl.cuh\>]{.preprocessor}
:::

::: {#astructbght_1_1cht_html .ttc}
::: ttname
[bght::cht](structbght_1_1cht.html)
:::

::: ttdoc
CHT CHT (cuckoo hash table) is an associative static GPU hash table that
contains key-value pairs wit\...
:::

::: ttdef
**Definition:** cht.hpp:49
:::
:::

::: {#astructbght_1_1cht_html_a090f8a449a06b62f43057d7a3fddea99 .ttc}
::: ttname
[bght::cht::operator=](structbght_1_1cht.html#a090f8a449a06b62f43057d7a3fddea99)
:::

::: ttdeci
cht & operator=(const cht &)=delete
:::

::: ttdoc
The assignment operator is currently deleted.
:::
:::

::: {#astructbght_1_1cht_html_a28b618c3f217d021318f097dab537c4a .ttc}
::: ttname
[bght::cht::clear](structbght_1_1cht.html#a28b618c3f217d021318f097dab537c4a)
:::

::: ttdeci
void clear()
:::

::: ttdoc
Clears the hash map and resets all slots.
:::
:::

::: {#astructbght_1_1cht_html_a33b47bb3d9abe89ab834e04b210f571c .ttc}
::: ttname
[bght::cht::insert](structbght_1_1cht.html#a33b47bb3d9abe89ab834e04b210f571c)
:::

::: ttdeci
bool insert(InputIt first, InputIt last, cudaStream_t stream=0)
:::

::: ttdoc
Host-side API for inserting all pairs defined by the input argument
iterators. All keys in the range \...
:::
:::

::: {#astructbght_1_1cht_html_a4e5e1879dbce693b73fd9569e9d86a34 .ttc}
::: ttname
[bght::cht::\~cht](structbght_1_1cht.html#a4e5e1879dbce693b73fd9569e9d86a34)
:::

::: ttdeci
\~cht()
:::

::: ttdoc
Destructor that destroys the hash map and deallocate memory if no copies
exist.
:::
:::

::: {#astructbght_1_1cht_html_a6f45a140a4f942230ffb4609096982ec .ttc}
::: ttname
[bght::cht::find](structbght_1_1cht.html#a6f45a140a4f942230ffb4609096982ec)
:::

::: ttdeci
\_\_device\_\_ mapped_type find(key_type const &key)
:::

::: ttdoc
Device-side cooperative find API that finds a single pair into the hash
map.
:::
:::

::: {#astructbght_1_1cht_html_a81fdda2b20c3dc2fd2c237e1e8c9ccac .ttc}
::: ttname
[bght::cht::find](structbght_1_1cht.html#a81fdda2b20c3dc2fd2c237e1e8c9ccac)
:::

::: ttdeci
void find(InputIt first, InputIt last, OutputIt output_begin,
cudaStream_t stream=0)
:::

::: ttdoc
Host-side API for finding all keys defined by the input argument
iterators.
:::
:::

::: {#astructbght_1_1cht_html_a87299241de2004dfe0bc7662401d5ea1 .ttc}
::: ttname
[bght::cht::cht](structbght_1_1cht.html#a87299241de2004dfe0bc7662401d5ea1)
:::

::: ttdeci
cht(cht &&)=delete
:::

::: ttdoc
Move constructor is currently deleted.
:::
:::

::: {#astructbght_1_1cht_html_a8a868596bd28694f9885d410ef3b37a3 .ttc}
::: ttname
[bght::cht::insert](structbght_1_1cht.html#a8a868596bd28694f9885d410ef3b37a3)
:::

::: ttdeci
\_\_device\_\_ bool insert(value_type const &pair)
:::

::: ttdoc
Device-side cooperative insertion API that inserts a single pair into
the hash map\....
:::
:::

::: {#astructbght_1_1cht_html_a9e7ca5b7d83f47e4188ffbeacbc4b7dc .ttc}
::: ttname
[bght::cht::randomize_hash_functions](structbght_1_1cht.html#a9e7ca5b7d83f47e4188ffbeacbc4b7dc)
:::

::: ttdeci
void randomize_hash_functions(RNG &rng)
:::

::: ttdoc
Host-side API to randomize the hash functions used for the probing
scheme. This can be used when the \...
:::
:::

::: {#astructbght_1_1cht_html_ab8475ae4c11257ff5476e6a5329a7af8 .ttc}
::: ttname
[bght::cht::cht](structbght_1_1cht.html#ab8475ae4c11257ff5476e6a5329a7af8)
:::

::: ttdeci
cht(const cht &other)
:::

::: ttdoc
A shallow-copy constructor.
:::
:::

::: {#astructbght_1_1cht_html_ad62008d5e9b37455c84620b908b88bdf .ttc}
::: ttname
[bght::cht::operator=](structbght_1_1cht.html#ad62008d5e9b37455c84620b908b88bdf)
:::

::: ttdeci
cht & operator=(cht &&)=delete
:::

::: ttdoc
The move assignment operator is currently deleted.
:::
:::

::: {#astructbght_1_1cht_html_aeba94f76492153267d6c2a8f0de331b5 .ttc}
::: ttname
[bght::cht::cht](structbght_1_1cht.html#aeba94f76492153267d6c2a8f0de331b5)
:::

::: ttdeci
cht(std::size_t capacity, Key sentinel_key, T sentinel_value, Allocator
const &allocator=Allocator{})
:::

::: ttdoc
Constructs the hash table with the specified capacity and uses the
specified sentinel key and value t\...
:::
:::

::: {#astructbght_1_1cuda__allocator_html .ttc}
::: ttname
[bght::cuda_allocator\< char \>](structbght_1_1cuda__allocator.html)
:::
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
