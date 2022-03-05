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
bcht.hpp
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
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00050}[ 50]{.lineno} [int]{.keywordtype} B = 16\>
:::

::: line
[]{#l00051}[ [51](structbght_1_1bcht.html){.line}]{.lineno}[struct
]{.keyword}[bcht](structbght_1_1bcht.html){.code .hl_struct} {
:::

::: line
[]{#l00052}[
[52](structbght_1_1bcht.html#a4b3cd889e1796dc5d4968c384a3532b2){.line}]{.lineno}
[using]{.keyword}
[value_type](structbght_1_1bcht.html#a4b3cd889e1796dc5d4968c384a3532b2){.code
.hl_typedef} = pair\<Key, T\>;
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
[80](structbght_1_1bcht.html#ab77ebf40b39673ff2943fbe950f6cee8){.line}]{.lineno}
[bcht](structbght_1_1bcht.html#ab77ebf40b39673ff2943fbe950f6cee8){.code
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
[87](structbght_1_1bcht.html#aab4f5aaf7d1d1c16581c63425c22d98f){.line}]{.lineno}
[bcht](structbght_1_1bcht.html#aab4f5aaf7d1d1c16581c63425c22d98f){.code
.hl_function}([const]{.keyword} [bcht](structbght_1_1bcht.html){.code
.hl_struct}& other);
:::

::: line
[]{#l00091}[
[91](structbght_1_1bcht.html#afdc822e8c97860d7d63b30ff66cd44a7){.line}]{.lineno}
[bcht](structbght_1_1bcht.html#afdc822e8c97860d7d63b30ff66cd44a7){.code
.hl_function}([bcht](structbght_1_1bcht.html){.code .hl_struct}&&) =
[delete]{.keyword};
:::

::: line
[]{#l00095}[
[95](structbght_1_1bcht.html#a2b193d98eb00b0c41900ff06cbb80fde){.line}]{.lineno}
[bcht](structbght_1_1bcht.html){.code .hl_struct}&
[operator=](structbght_1_1bcht.html#a2b193d98eb00b0c41900ff06cbb80fde){.code
.hl_function}([const]{.keyword} [bcht](structbght_1_1bcht.html){.code
.hl_struct}&) = [delete]{.keyword};
:::

::: line
[]{#l00099}[
[99](structbght_1_1bcht.html#ab91884025ae177f74ad6938789c23f3a){.line}]{.lineno}
[bcht](structbght_1_1bcht.html){.code .hl_struct}&
[operator=](structbght_1_1bcht.html#ab91884025ae177f74ad6938789c23f3a){.code
.hl_function}([bcht](structbght_1_1bcht.html){.code .hl_struct}&&) =
[delete]{.keyword};
:::

::: line
[]{#l00103}[
[103](structbght_1_1bcht.html#a99a897955e80d66cb83d62ef33d79110){.line}]{.lineno}
[\~bcht](structbght_1_1bcht.html#a99a897955e80d66cb83d62ef33d79110){.code
.hl_function}();
:::

::: line
[]{#l00104}[ 104]{.lineno}
:::

::: line
[]{#l00108}[
[108](structbght_1_1bcht.html#a798fc4be13d418089ac8d491e98b5147){.line}]{.lineno}
[void]{.keywordtype}
[clear](structbght_1_1bcht.html#a798fc4be13d418089ac8d491e98b5147){.code
.hl_function}();
:::

::: line
[]{#l00109}[ 109]{.lineno}
:::

::: line
[]{#l00120}[ 120]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt\>
:::

::: line
[]{#l00121}[
[121](structbght_1_1bcht.html#a2e8e21183a7978f2908a801f98138d22){.line}]{.lineno}
[bool]{.keywordtype}
[insert](structbght_1_1bcht.html#a2e8e21183a7978f2908a801f98138d22){.code
.hl_function}(InputIt first, InputIt last, cudaStream_t stream = 0);
:::

::: line
[]{#l00122}[ 122]{.lineno}
:::

::: line
[]{#l00134}[ 134]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt\>
:::

::: line
[]{#l00135}[
[135](structbght_1_1bcht.html#aa00dd13fd68144e59cfdd1bd0ce212dc){.line}]{.lineno}
[void]{.keywordtype}
[find](structbght_1_1bcht.html#aa00dd13fd68144e59cfdd1bd0ce212dc){.code
.hl_function}(InputIt first, InputIt last, OutputIt output_begin,
cudaStream_t stream = 0);
:::

::: line
[]{#l00136}[ 136]{.lineno}
:::

::: line
[]{#l00149}[ 149]{.lineno} [template]{.keyword} \<[typename]{.keyword}
tile_type\>
:::

::: line
[]{#l00150}[
[150](structbght_1_1bcht.html#a8ca06cef309ce158a21ba686088a5804){.line}]{.lineno}
\_\_device\_\_ [bool]{.keywordtype}
[insert](structbght_1_1bcht.html#a8ca06cef309ce158a21ba686088a5804){.code
.hl_function}([value_type](structbght_1_1bcht.html#a4b3cd889e1796dc5d4968c384a3532b2){.code
.hl_typedef} [const]{.keyword}& pair, tile_type [const]{.keyword}&
tile);
:::

::: line
[]{#l00151}[ 151]{.lineno}
:::

::: line
[]{#l00164}[ 164]{.lineno} [template]{.keyword} \<[typename]{.keyword}
tile_type\>
:::

::: line
[]{#l00165}[
[165](structbght_1_1bcht.html#adbbadf9704bd5752b55fe0e7ab9d788c){.line}]{.lineno}
\_\_device\_\_ mapped_type
[find](structbght_1_1bcht.html#adbbadf9704bd5752b55fe0e7ab9d788c){.code
.hl_function}(key_type [const]{.keyword}& key, tile_type
[const]{.keyword}& tile);
:::

::: line
[]{#l00166}[ 166]{.lineno}
:::

::: line
[]{#l00174}[ 174]{.lineno} [template]{.keyword} \<[typename]{.keyword}
RNG\>
:::

::: line
[]{#l00175}[
[175](structbght_1_1bcht.html#afceb31cd69d17b0051edd59550babd23){.line}]{.lineno}
[void]{.keywordtype}
[randomize_hash_functions](structbght_1_1bcht.html#afceb31cd69d17b0051edd59550babd23){.code
.hl_function}(RNG& rng);
:::

::: line
[]{#l00176}[ 176]{.lineno}
:::

::: line
[]{#l00181}[
[181](structbght_1_1bcht.html#a4fc21878958c178a66d686039c18f957){.line}]{.lineno}
size_type
[size](structbght_1_1bcht.html#a4fc21878958c178a66d686039c18f957){.code
.hl_function}(cudaStream_t stream = 0);
:::

::: line
[]{#l00182}[ 182]{.lineno}
:::

::: line
[]{#l00183}[ 183]{.lineno} [private]{.keyword}:
:::

::: line
[]{#l00184}[ 184]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00185}[ 185]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::tiled_insert_kernel(InputIt,
InputIt, HashMap);
:::

::: line
[]{#l00186}[ 186]{.lineno}
:::

::: line
[]{#l00187}[ 187]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00188}[ 188]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::tiled_find_kernel(InputIt,
:::

::: line
[]{#l00189}[ 189]{.lineno} InputIt,
:::

::: line
[]{#l00190}[ 190]{.lineno} OutputIt,
:::

::: line
[]{#l00191}[ 191]{.lineno} HashMap);
:::

::: line
[]{#l00192}[ 192]{.lineno}
:::

::: line
[]{#l00193}[ 193]{.lineno} [template]{.keyword} \<[int]{.keywordtype}
BlockSize, [typename]{.keyword} InputT, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00194}[ 194]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::count_kernel([const]{.keyword}
InputT,
:::

::: line
[]{#l00195}[ 195]{.lineno} std::size_t\*,
:::

::: line
[]{#l00196}[ 196]{.lineno} HashMap);
:::

::: line
[]{#l00197}[ 197]{.lineno}
:::

::: line
[]{#l00198}[ 198]{.lineno} std::size_t capacity\_;
:::

::: line
[]{#l00199}[ 199]{.lineno} key_type sentinel_key\_{};
:::

::: line
[]{#l00200}[ 200]{.lineno} mapped_type sentinel_value\_{};
:::

::: line
[]{#l00201}[ 201]{.lineno} allocator_type allocator\_;
:::

::: line
[]{#l00202}[ 202]{.lineno} atomic_pair_allocator_type
atomic_pairs_allocator\_;
:::

::: line
[]{#l00203}[ 203]{.lineno} pool_allocator_type pool_allocator\_;
:::

::: line
[]{#l00204}[ 204]{.lineno} size_type_allocator_type
size_type_allocator\_;
:::

::: line
[]{#l00205}[ 205]{.lineno}
:::

::: line
[]{#l00206}[ 206]{.lineno} atomic_pair_type\* d_table\_{};
:::

::: line
[]{#l00207}[ 207]{.lineno} std::shared_ptr\<atomic_pair_type\> table\_;
:::

::: line
[]{#l00208}[ 208]{.lineno}
:::

::: line
[]{#l00209}[ 209]{.lineno} [bool]{.keywordtype}\* d_build_success\_;
:::

::: line
[]{#l00210}[ 210]{.lineno} std::shared_ptr\<bool\> build_success\_;
:::

::: line
[]{#l00211}[ 211]{.lineno}
:::

::: line
[]{#l00212}[ 212]{.lineno} uint32_t max_cuckoo_chains\_;
:::

::: line
[]{#l00213}[ 213]{.lineno}
:::

::: line
[]{#l00214}[ 214]{.lineno} Hash hf0\_;
:::

::: line
[]{#l00215}[ 215]{.lineno} Hash hf1\_;
:::

::: line
[]{#l00216}[ 216]{.lineno} Hash hf2\_;
:::

::: line
[]{#l00217}[ 217]{.lineno}
:::

::: line
[]{#l00218}[ 218]{.lineno} std::size_t num_buckets\_;
:::

::: line
[]{#l00219}[ 219]{.lineno}};
:::

::: line
[]{#l00220}[ 220]{.lineno}
:::

::: line
[]{#l00221}[ 221]{.lineno}} [// namespace bght]{.comment}
:::

::: line
[]{#l00222}[ 222]{.lineno}
:::

::: line
[]{#l00223}[ 223]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T\>
:::

::: line
[]{#l00224}[ 224]{.lineno}[using]{.keyword} bcht8 = [typename]{.keyword}
[bght::bcht](structbght_1_1bcht.html){.code .hl_struct}\<Key,
:::

::: line
[]{#l00225}[ 225]{.lineno} T,
:::

::: line
[]{#l00226}[ 226]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00227}[ 227]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00228}[ 228]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00229}[ 229]{.lineno}
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00230}[ 230]{.lineno} 8\>;
:::

::: line
[]{#l00231}[ 231]{.lineno}
:::

::: line
[]{#l00232}[ 232]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T\>
:::

::: line
[]{#l00233}[ 233]{.lineno}[using]{.keyword} bcht16 =
[typename]{.keyword} [bght::bcht](structbght_1_1bcht.html){.code
.hl_struct}\<Key,
:::

::: line
[]{#l00234}[ 234]{.lineno} T,
:::

::: line
[]{#l00235}[ 235]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00236}[ 236]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00237}[ 237]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00238}[ 238]{.lineno}
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00239}[ 239]{.lineno} 16\>;
:::

::: line
[]{#l00240}[ 240]{.lineno}
:::

::: line
[]{#l00241}[ 241]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T\>
:::

::: line
[]{#l00242}[ 242]{.lineno}[using]{.keyword} bcht32 =
[typename]{.keyword} [bght::bcht](structbght_1_1bcht.html){.code
.hl_struct}\<Key,
:::

::: line
[]{#l00243}[ 243]{.lineno} T,
:::

::: line
[]{#l00244}[ 244]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00245}[ 245]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00246}[ 246]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00247}[ 247]{.lineno}
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00248}[ 248]{.lineno} 32\>;
:::

::: line
[]{#l00249}[ 249]{.lineno}
:::

::: line
[]{#l00250}[ 250]{.lineno}[#include
\<detail/bcht_impl.cuh\>]{.preprocessor}
:::

::: {#astructbght_1_1bcht_html .ttc}
::: ttname
[bght::bcht](structbght_1_1bcht.html)
:::

::: ttdoc
BCHT BCHT (bucketed cuckoo hash table) is an associative static GPU hash
table that contains key-valu\...
:::

::: ttdef
**Definition:** bcht.hpp:51
:::
:::

::: {#astructbght_1_1bcht_html_a2b193d98eb00b0c41900ff06cbb80fde .ttc}
::: ttname
[bght::bcht::operator=](structbght_1_1bcht.html#a2b193d98eb00b0c41900ff06cbb80fde)
:::

::: ttdeci
bcht & operator=(const bcht &)=delete
:::

::: ttdoc
The assignment operator for the BCHT is currently deleted.
:::
:::

::: {#astructbght_1_1bcht_html_a2e8e21183a7978f2908a801f98138d22 .ttc}
::: ttname
[bght::bcht::insert](structbght_1_1bcht.html#a2e8e21183a7978f2908a801f98138d22)
:::

::: ttdeci
bool insert(InputIt first, InputIt last, cudaStream_t stream=0)
:::

::: ttdoc
Host-side API for inserting all pairs defined by the input argument
iterators. All keys in the range \...
:::
:::

::: {#astructbght_1_1bcht_html_a4b3cd889e1796dc5d4968c384a3532b2 .ttc}
::: ttname
[bght::bcht::value_type](structbght_1_1bcht.html#a4b3cd889e1796dc5d4968c384a3532b2)
:::

::: ttdeci
pair\< Key, T \> value_type
:::

::: ttdoc
Pair type.
:::

::: ttdef
**Definition:** bcht.hpp:52
:::
:::

::: {#astructbght_1_1bcht_html_a4fc21878958c178a66d686039c18f957 .ttc}
::: ttname
[bght::bcht::size](structbght_1_1bcht.html#a4fc21878958c178a66d686039c18f957)
:::

::: ttdeci
size_type size(cudaStream_t stream=0)
:::

::: ttdoc
Compute the number of elements in the map.
:::
:::

::: {#astructbght_1_1bcht_html_a798fc4be13d418089ac8d491e98b5147 .ttc}
::: ttname
[bght::bcht::clear](structbght_1_1bcht.html#a798fc4be13d418089ac8d491e98b5147)
:::

::: ttdeci
void clear()
:::

::: ttdoc
Clears the hash map and resets all slots.
:::
:::

::: {#astructbght_1_1bcht_html_a8ca06cef309ce158a21ba686088a5804 .ttc}
::: ttname
[bght::bcht::insert](structbght_1_1bcht.html#a8ca06cef309ce158a21ba686088a5804)
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

::: {#astructbght_1_1bcht_html_a99a897955e80d66cb83d62ef33d79110 .ttc}
::: ttname
[bght::bcht::\~bcht](structbght_1_1bcht.html#a99a897955e80d66cb83d62ef33d79110)
:::

::: ttdeci
\~bcht()
:::

::: ttdoc
Destructor that destroys the hash map and deallocate memory if no copies
exist.
:::
:::

::: {#astructbght_1_1bcht_html_aa00dd13fd68144e59cfdd1bd0ce212dc .ttc}
::: ttname
[bght::bcht::find](structbght_1_1bcht.html#aa00dd13fd68144e59cfdd1bd0ce212dc)
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

::: {#astructbght_1_1bcht_html_aab4f5aaf7d1d1c16581c63425c22d98f .ttc}
::: ttname
[bght::bcht::bcht](structbght_1_1bcht.html#aab4f5aaf7d1d1c16581c63425c22d98f)
:::

::: ttdeci
bcht(const bcht &other)
:::

::: ttdoc
A shallow-copy constructor.
:::
:::

::: {#astructbght_1_1bcht_html_ab77ebf40b39673ff2943fbe950f6cee8 .ttc}
::: ttname
[bght::bcht::bcht](structbght_1_1bcht.html#ab77ebf40b39673ff2943fbe950f6cee8)
:::

::: ttdeci
bcht(std::size_t capacity, Key sentinel_key, T sentinel_value, Allocator
const &allocator=Allocator{})
:::

::: ttdoc
Constructs the hash table with the specified capacity and uses the
specified sentinel key and value t\...
:::
:::

::: {#astructbght_1_1bcht_html_ab91884025ae177f74ad6938789c23f3a .ttc}
::: ttname
[bght::bcht::operator=](structbght_1_1bcht.html#ab91884025ae177f74ad6938789c23f3a)
:::

::: ttdeci
bcht & operator=(bcht &&)=delete
:::

::: ttdoc
The move assignment operator for the BCHT is currently deleted.
:::
:::

::: {#astructbght_1_1bcht_html_adbbadf9704bd5752b55fe0e7ab9d788c .ttc}
::: ttname
[bght::bcht::find](structbght_1_1bcht.html#adbbadf9704bd5752b55fe0e7ab9d788c)
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

::: {#astructbght_1_1bcht_html_afceb31cd69d17b0051edd59550babd23 .ttc}
::: ttname
[bght::bcht::randomize_hash_functions](structbght_1_1bcht.html#afceb31cd69d17b0051edd59550babd23)
:::

::: ttdeci
void randomize_hash_functions(RNG &rng)
:::

::: ttdoc
Host-side API to randomize the hash functions used for the probing
scheme. This can be used when the \...
:::
:::

::: {#astructbght_1_1bcht_html_afdc822e8c97860d7d63b30ff66cd44a7 .ttc}
::: ttname
[bght::bcht::bcht](structbght_1_1bcht.html#afdc822e8c97860d7d63b30ff66cd44a7)
:::

::: ttdeci
bcht(bcht &&)=delete
:::

::: ttdoc
Move constructor is currently deleted.
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
