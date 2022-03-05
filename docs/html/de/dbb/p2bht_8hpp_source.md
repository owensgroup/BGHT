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
-   [include](../../dir_d44c64559bbebec7f509842c48db8b23.html){.el}
:::
:::

::: header
::: headertitle
::: title
p2bht.hpp
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
[]{#l00044}[ 44]{.lineno}[template]{.keyword} \<[class ]{.keyword}Key,
:::

::: line
[]{#l00045}[ 45]{.lineno} [class ]{.keyword}T,
:::

::: line
[]{#l00046}[ 46]{.lineno} [class ]{.keyword}Hash =
bght::universal_hash\<Key\>,
:::

::: line
[]{#l00047}[ 47]{.lineno} [class ]{.keyword}KeyEqual =
bght::equal_to\<Key\>,
:::

::: line
[]{#l00048}[ 48]{.lineno} cuda::thread_scope Scope =
cuda::thread_scope_device,
:::

::: line
[]{#l00049}[ 49]{.lineno} [class ]{.keyword}Allocator =
[bght::cuda_allocator\<char\>](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00050}[ 50]{.lineno} [int]{.keywordtype} B = 16\>
:::

::: line
[]{#l00051}[
[51](../../d6/dcc/structbght_1_1p2bht.html){.line}]{.lineno}[struct
]{.keyword}[p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code
.hl_struct} {
:::

::: line
[]{#l00052}[ 52]{.lineno} [using]{.keyword} value_type = pair\<Key, T\>;
:::

::: line
[]{#l00053}[ 53]{.lineno} [using]{.keyword} key_type = Key;
:::

::: line
[]{#l00054}[ 54]{.lineno} [using]{.keyword} mapped_type = T;
:::

::: line
[]{#l00055}[ 55]{.lineno} [using]{.keyword} atomic_pair_type =
cuda::atomic\<value_type, Scope\>;
:::

::: line
[]{#l00056}[ 56]{.lineno} [using]{.keyword} allocator_type = Allocator;
:::

::: line
[]{#l00057}[ 57]{.lineno} [using]{.keyword} hasher = Hash;
:::

::: line
[]{#l00058}[ 58]{.lineno} [using]{.keyword} size_type = std::size_t;
:::

::: line
[]{#l00059}[ 59]{.lineno}
:::

::: line
[]{#l00060}[ 60]{.lineno} [using]{.keyword} atomic_pair_allocator_type =
:::

::: line
[]{#l00061}[ 61]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<atomic_pair_type\>;
:::

::: line
[]{#l00062}[ 62]{.lineno} [using]{.keyword} pool_allocator_type =
:::

::: line
[]{#l00063}[ 63]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<[bool]{.keywordtype}\>;
:::

::: line
[]{#l00064}[ 64]{.lineno} [using]{.keyword} size_type_allocator_type =
:::

::: line
[]{#l00065}[ 65]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<size_type\>;
:::

::: line
[]{#l00066}[ 66]{.lineno}
:::

::: line
[]{#l00067}[ 67]{.lineno} [static]{.keyword} [constexpr]{.keyword}
[auto]{.keyword} bucket_size = B;
:::

::: line
[]{#l00068}[ 68]{.lineno} [using]{.keyword} key_equal = KeyEqual;
:::

::: line
[]{#l00069}[ 69]{.lineno}
:::

::: line
[]{#l00080}[
[80](../../d6/dcc/structbght_1_1p2bht.html#a9e57cde3bed4b452a1699802f1390b58){.line}]{.lineno}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html#a9e57cde3bed4b452a1699802f1390b58){.code
.hl_function}(std::size_t capacity,
:::

::: line
[]{#l00081}[ 81]{.lineno} Key sentinel_key,
:::

::: line
[]{#l00082}[ 82]{.lineno} T sentinel_value,
:::

::: line
[]{#l00083}[ 83]{.lineno} Allocator [const]{.keyword}& allocator =
Allocator{});
:::

::: line
[]{#l00087}[
[87](../../d6/dcc/structbght_1_1p2bht.html#a5b86e49577c0fec06aed5c6990cf93b9){.line}]{.lineno}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html#a5b86e49577c0fec06aed5c6990cf93b9){.code
.hl_function}([const]{.keyword}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code .hl_struct}&
other);
:::

::: line
[]{#l00091}[
[91](../../d6/dcc/structbght_1_1p2bht.html#afd4542b3692395f6af3ef6a214fc0001){.line}]{.lineno}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html#afd4542b3692395f6af3ef6a214fc0001){.code
.hl_function}([p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code
.hl_struct}&&) = [delete]{.keyword};
:::

::: line
[]{#l00095}[
[95](../../d6/dcc/structbght_1_1p2bht.html#a447e9bbf421c9d6a8983da989629994d){.line}]{.lineno}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code .hl_struct}&
[operator=](../../d6/dcc/structbght_1_1p2bht.html#a447e9bbf421c9d6a8983da989629994d){.code
.hl_function}([const]{.keyword}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code .hl_struct}&) =
[delete]{.keyword};
:::

::: line
[]{#l00099}[
[99](../../d6/dcc/structbght_1_1p2bht.html#af71794cec02a46adc02a875ca2048899){.line}]{.lineno}
[p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code .hl_struct}&
[operator=](../../d6/dcc/structbght_1_1p2bht.html#af71794cec02a46adc02a875ca2048899){.code
.hl_function}([p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code
.hl_struct}&&) = [delete]{.keyword};
:::

::: line
[]{#l00103}[
[103](../../d6/dcc/structbght_1_1p2bht.html#aee3fe77f75a7825a060d9b891726cff3){.line}]{.lineno}
[\~p2bht](../../d6/dcc/structbght_1_1p2bht.html#aee3fe77f75a7825a060d9b891726cff3){.code
.hl_function}();
:::

::: line
[]{#l00107}[
[107](../../d6/dcc/structbght_1_1p2bht.html#ab2181363291624ae8b152a4438ff6caf){.line}]{.lineno}
[void]{.keywordtype}
[clear](../../d6/dcc/structbght_1_1p2bht.html#ab2181363291624ae8b152a4438ff6caf){.code
.hl_function}();
:::

::: line
[]{#l00108}[ 108]{.lineno}
:::

::: line
[]{#l00119}[ 119]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt\>
:::

::: line
[]{#l00120}[
[120](../../d6/dcc/structbght_1_1p2bht.html#afaf41c013493f4912537f20ec716f015){.line}]{.lineno}
[bool]{.keywordtype}
[insert](../../d6/dcc/structbght_1_1p2bht.html#afaf41c013493f4912537f20ec716f015){.code
.hl_function}(InputIt first, InputIt last, cudaStream_t stream = 0);
:::

::: line
[]{#l00121}[ 121]{.lineno}
:::

::: line
[]{#l00133}[ 133]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt\>
:::

::: line
[]{#l00134}[
[134](../../d6/dcc/structbght_1_1p2bht.html#a1d024b93ef3392e9634b1160ab679d92){.line}]{.lineno}
[void]{.keywordtype}
[find](../../d6/dcc/structbght_1_1p2bht.html#a1d024b93ef3392e9634b1160ab679d92){.code
.hl_function}(InputIt first, InputIt last, OutputIt output_begin,
cudaStream_t stream = 0);
:::

::: line
[]{#l00135}[ 135]{.lineno}
:::

::: line
[]{#l00148}[ 148]{.lineno} [template]{.keyword} \<[typename]{.keyword}
tile_type\>
:::

::: line
[]{#l00149}[
[149](../../d6/dcc/structbght_1_1p2bht.html#ad6bcca773f08880cdfe156352b8925d8){.line}]{.lineno}
\_\_device\_\_ [bool]{.keywordtype}
[insert](../../d6/dcc/structbght_1_1p2bht.html#ad6bcca773f08880cdfe156352b8925d8){.code
.hl_function}(value_type [const]{.keyword}& pair, tile_type
[const]{.keyword}& tile);
:::

::: line
[]{#l00150}[ 150]{.lineno}
:::

::: line
[]{#l00163}[ 163]{.lineno} [template]{.keyword} \<[typename]{.keyword}
tile_type\>
:::

::: line
[]{#l00164}[
[164](../../d6/dcc/structbght_1_1p2bht.html#a913cca5c95d7b9c7ad5a61ea6769e9d6){.line}]{.lineno}
\_\_device\_\_ mapped_type
[find](../../d6/dcc/structbght_1_1p2bht.html#a913cca5c95d7b9c7ad5a61ea6769e9d6){.code
.hl_function}(key_type [const]{.keyword}& key, tile_type
[const]{.keyword}& tile);
:::

::: line
[]{#l00165}[ 165]{.lineno}
:::

::: line
[]{#l00173}[ 173]{.lineno} [template]{.keyword} \<[typename]{.keyword}
RNG\>
:::

::: line
[]{#l00174}[
[174](../../d6/dcc/structbght_1_1p2bht.html#a9112b518945f384f948ed52aa0c5ad80){.line}]{.lineno}
[void]{.keywordtype}
[randomize_hash_functions](../../d6/dcc/structbght_1_1p2bht.html#a9112b518945f384f948ed52aa0c5ad80){.code
.hl_function}(RNG& rng);
:::

::: line
[]{#l00175}[ 175]{.lineno}
:::

::: line
[]{#l00180}[
[180](../../d6/dcc/structbght_1_1p2bht.html#a4f95a1f7a39e693a0ca1cd67951c1ec7){.line}]{.lineno}
size_type
[size](../../d6/dcc/structbght_1_1p2bht.html#a4f95a1f7a39e693a0ca1cd67951c1ec7){.code
.hl_function}(cudaStream_t stream = 0);
:::

::: line
[]{#l00181}[ 181]{.lineno}
:::

::: line
[]{#l00182}[ 182]{.lineno} [private]{.keyword}:
:::

::: line
[]{#l00183}[ 183]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00184}[ 184]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::tiled_insert_kernel(InputIt,
InputIt, HashMap);
:::

::: line
[]{#l00185}[ 185]{.lineno}
:::

::: line
[]{#l00186}[ 186]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00187}[ 187]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::tiled_find_kernel(InputIt,
:::

::: line
[]{#l00188}[ 188]{.lineno} InputIt,
:::

::: line
[]{#l00189}[ 189]{.lineno} OutputIt,
:::

::: line
[]{#l00190}[ 190]{.lineno} HashMap);
:::

::: line
[]{#l00191}[ 191]{.lineno}
:::

::: line
[]{#l00192}[ 192]{.lineno} [template]{.keyword} \<[int]{.keywordtype}
BlockSize, [typename]{.keyword} InputT, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00193}[ 193]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::count_kernel([const]{.keyword}
InputT,
:::

::: line
[]{#l00194}[ 194]{.lineno} std::size_t\*,
:::

::: line
[]{#l00195}[ 195]{.lineno} HashMap);
:::

::: line
[]{#l00196}[ 196]{.lineno}
:::

::: line
[]{#l00197}[ 197]{.lineno} std::size_t capacity\_;
:::

::: line
[]{#l00198}[ 198]{.lineno} key_type sentinel_key\_{};
:::

::: line
[]{#l00199}[ 199]{.lineno} mapped_type sentinel_value\_{};
:::

::: line
[]{#l00200}[ 200]{.lineno} allocator_type allocator\_;
:::

::: line
[]{#l00201}[ 201]{.lineno} atomic_pair_allocator_type
atomic_pairs_allocator\_;
:::

::: line
[]{#l00202}[ 202]{.lineno} pool_allocator_type pool_allocator\_;
:::

::: line
[]{#l00203}[ 203]{.lineno} size_type_allocator_type
size_type_allocator\_;
:::

::: line
[]{#l00204}[ 204]{.lineno}
:::

::: line
[]{#l00205}[ 205]{.lineno} atomic_pair_type\* d_table\_{};
:::

::: line
[]{#l00206}[ 206]{.lineno} std::shared_ptr\<atomic_pair_type\> table\_;
:::

::: line
[]{#l00207}[ 207]{.lineno}
:::

::: line
[]{#l00208}[ 208]{.lineno} [bool]{.keywordtype}\* d_build_success\_;
:::

::: line
[]{#l00209}[ 209]{.lineno} std::shared_ptr\<bool\> build_success\_;
:::

::: line
[]{#l00210}[ 210]{.lineno}
:::

::: line
[]{#l00211}[ 211]{.lineno} Hash hf0\_;
:::

::: line
[]{#l00212}[ 212]{.lineno} Hash hf1\_;
:::

::: line
[]{#l00213}[ 213]{.lineno}
:::

::: line
[]{#l00214}[ 214]{.lineno} std::size_t num_buckets\_;
:::

::: line
[]{#l00215}[ 215]{.lineno}};
:::

::: line
[]{#l00216}[ 216]{.lineno}
:::

::: line
[]{#l00217}[ 217]{.lineno}} [// namespace bght]{.comment}
:::

::: line
[]{#l00218}[ 218]{.lineno}
:::

::: line
[]{#l00219}[ 219]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T\>
:::

::: line
[]{#l00220}[ 220]{.lineno}[using]{.keyword} p2bht8 =
[typename]{.keyword}
[bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code
.hl_struct}\<Key,
:::

::: line
[]{#l00221}[ 221]{.lineno} T,
:::

::: line
[]{#l00222}[ 222]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00223}[ 223]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00224}[ 224]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00225}[ 225]{.lineno}
[bght::cuda_allocator\<char\>](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00226}[ 226]{.lineno} 8\>;
:::

::: line
[]{#l00227}[ 227]{.lineno}
:::

::: line
[]{#l00228}[ 228]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T\>
:::

::: line
[]{#l00229}[ 229]{.lineno}[using]{.keyword} p2bht16 =
[typename]{.keyword}
[bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code
.hl_struct}\<Key,
:::

::: line
[]{#l00230}[ 230]{.lineno} T,
:::

::: line
[]{#l00231}[ 231]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00232}[ 232]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00233}[ 233]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00234}[ 234]{.lineno}
[bght::cuda_allocator\<char\>](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00235}[ 235]{.lineno} 16\>;
:::

::: line
[]{#l00236}[ 236]{.lineno}
:::

::: line
[]{#l00237}[ 237]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T\>
:::

::: line
[]{#l00238}[ 238]{.lineno}[using]{.keyword} p2bht32 =
[typename]{.keyword}
[bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html){.code
.hl_struct}\<Key,
:::

::: line
[]{#l00239}[ 239]{.lineno} T,
:::

::: line
[]{#l00240}[ 240]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00241}[ 241]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00242}[ 242]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00243}[ 243]{.lineno}
[bght::cuda_allocator\<char\>](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00244}[ 244]{.lineno} 32\>;
:::

::: line
[]{#l00245}[ 245]{.lineno}
:::

::: line
[]{#l00246}[ 246]{.lineno}[#include
\<detail/p2bht_impl.cuh\>]{.preprocessor}
:::

::: {#astructbght_1_1cuda__allocator_html .ttc}
::: ttname
[bght::cuda_allocator\< char
\>](../../d1/df4/structbght_1_1cuda__allocator.html)
:::
:::

::: {#astructbght_1_1p2bht_html .ttc}
::: ttname
[bght::p2bht](../../d6/dcc/structbght_1_1p2bht.html)
:::

::: ttdoc
P2BHT P2BHT (power-of-two bucketed hash table) is an associative static
GPU hash table that contains \...
:::

::: ttdef
**Definition:** p2bht.hpp:51
:::
:::

::: {#astructbght_1_1p2bht_html_a1d024b93ef3392e9634b1160ab679d92 .ttc}
::: ttname
[bght::p2bht::find](../../d6/dcc/structbght_1_1p2bht.html#a1d024b93ef3392e9634b1160ab679d92)
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

::: {#astructbght_1_1p2bht_html_a447e9bbf421c9d6a8983da989629994d .ttc}
::: ttname
[bght::p2bht::operator=](../../d6/dcc/structbght_1_1p2bht.html#a447e9bbf421c9d6a8983da989629994d)
:::

::: ttdeci
p2bht & operator=(const p2bht &)=delete
:::

::: ttdoc
The assignment operator is currently deleted.
:::
:::

::: {#astructbght_1_1p2bht_html_a4f95a1f7a39e693a0ca1cd67951c1ec7 .ttc}
::: ttname
[bght::p2bht::size](../../d6/dcc/structbght_1_1p2bht.html#a4f95a1f7a39e693a0ca1cd67951c1ec7)
:::

::: ttdeci
size_type size(cudaStream_t stream=0)
:::

::: ttdoc
Compute the number of elements in the map.
:::
:::

::: {#astructbght_1_1p2bht_html_a5b86e49577c0fec06aed5c6990cf93b9 .ttc}
::: ttname
[bght::p2bht::p2bht](../../d6/dcc/structbght_1_1p2bht.html#a5b86e49577c0fec06aed5c6990cf93b9)
:::

::: ttdeci
p2bht(const p2bht &other)
:::

::: ttdoc
A shallow-copy constructor.
:::
:::

::: {#astructbght_1_1p2bht_html_a9112b518945f384f948ed52aa0c5ad80 .ttc}
::: ttname
[bght::p2bht::randomize_hash_functions](../../d6/dcc/structbght_1_1p2bht.html#a9112b518945f384f948ed52aa0c5ad80)
:::

::: ttdeci
void randomize_hash_functions(RNG &rng)
:::

::: ttdoc
Host-side API to randomize the hash functions used for the probing
scheme. This can be used when the \...
:::
:::

::: {#astructbght_1_1p2bht_html_a913cca5c95d7b9c7ad5a61ea6769e9d6 .ttc}
::: ttname
[bght::p2bht::find](../../d6/dcc/structbght_1_1p2bht.html#a913cca5c95d7b9c7ad5a61ea6769e9d6)
:::

::: ttdeci
\_\_device\_\_ mapped_type find(key_type const &key, tile_type const
&tile)
:::

::: ttdoc
Device-side cooperative find API that finds a single pair into the hash
map.
:::
:::

::: {#astructbght_1_1p2bht_html_a9e57cde3bed4b452a1699802f1390b58 .ttc}
::: ttname
[bght::p2bht::p2bht](../../d6/dcc/structbght_1_1p2bht.html#a9e57cde3bed4b452a1699802f1390b58)
:::

::: ttdeci
p2bht(std::size_t capacity, Key sentinel_key, T sentinel_value,
Allocator const &allocator=Allocator{})
:::

::: ttdoc
Constructs the hash table with the specified capacity and uses the
specified sentinel key and value t\...
:::
:::

::: {#astructbght_1_1p2bht_html_ab2181363291624ae8b152a4438ff6caf .ttc}
::: ttname
[bght::p2bht::clear](../../d6/dcc/structbght_1_1p2bht.html#ab2181363291624ae8b152a4438ff6caf)
:::

::: ttdeci
void clear()
:::

::: ttdoc
Clears the hash map and resets all slots.
:::
:::

::: {#astructbght_1_1p2bht_html_ad6bcca773f08880cdfe156352b8925d8 .ttc}
::: ttname
[bght::p2bht::insert](../../d6/dcc/structbght_1_1p2bht.html#ad6bcca773f08880cdfe156352b8925d8)
:::

::: ttdeci
\_\_device\_\_ bool insert(value_type const &pair, tile_type const
&tile)
:::

::: ttdoc
Device-side cooperative insertion API that inserts a single pair into
the hash map.
:::
:::

::: {#astructbght_1_1p2bht_html_aee3fe77f75a7825a060d9b891726cff3 .ttc}
::: ttname
[bght::p2bht::\~p2bht](../../d6/dcc/structbght_1_1p2bht.html#aee3fe77f75a7825a060d9b891726cff3)
:::

::: ttdeci
\~p2bht()
:::

::: ttdoc
Destructor that destroys the hash map and deallocate memory if no copies
exist.
:::
:::

::: {#astructbght_1_1p2bht_html_af71794cec02a46adc02a875ca2048899 .ttc}
::: ttname
[bght::p2bht::operator=](../../d6/dcc/structbght_1_1p2bht.html#af71794cec02a46adc02a875ca2048899)
:::

::: ttdeci
p2bht & operator=(p2bht &&)=delete
:::

::: ttdoc
The move assignment operator is currently deleted.
:::
:::

::: {#astructbght_1_1p2bht_html_afaf41c013493f4912537f20ec716f015 .ttc}
::: ttname
[bght::p2bht::insert](../../d6/dcc/structbght_1_1p2bht.html#afaf41c013493f4912537f20ec716f015)
:::

::: ttdeci
bool insert(InputIt first, InputIt last, cudaStream_t stream=0)
:::

::: ttdoc
Host-side API for inserting all pairs defined by the input argument
iterators. All keys in the range \...
:::
:::

::: {#astructbght_1_1p2bht_html_afd4542b3692395f6af3ef6a214fc0001 .ttc}
::: ttname
[bght::p2bht::p2bht](../../d6/dcc/structbght_1_1p2bht.html#afd4542b3692395f6af3ef6a214fc0001)
:::

::: ttdeci
p2bht(p2bht &&)=delete
:::

::: ttdoc
Move constructor is currently deleted.
:::
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
