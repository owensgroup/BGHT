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
iht.hpp
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
[]{#l00019}[ 19]{.lineno}[#include
\<detail/allocator.hpp\>]{.preprocessor}
:::

::: line
[]{#l00020}[ 20]{.lineno}[#include
\<detail/cuda_helpers.cuh\>]{.preprocessor}
:::

::: line
[]{#l00021}[ 21]{.lineno}[#include
\<detail/hash_functions.cuh\>]{.preprocessor}
:::

::: line
[]{#l00022}[ 22]{.lineno}[#include
\<detail/kernels.cuh\>]{.preprocessor}
:::

::: line
[]{#l00023}[ 23]{.lineno}[#include \<detail/pair.cuh\>]{.preprocessor}
:::

::: line
[]{#l00024}[ 24]{.lineno}[#include \<memory\>]{.preprocessor}
:::

::: line
[]{#l00025}[ 25]{.lineno}
:::

::: line
[]{#l00026}[ 26]{.lineno}[namespace ]{.keyword}bght {
:::

::: line
[]{#l00027}[ 27]{.lineno}
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
[]{#l00050}[ 50]{.lineno} [int]{.keywordtype} B = 16,
:::

::: line
[]{#l00051}[ 51]{.lineno} [int]{.keywordtype} Threshold = 14\>
:::

::: line
[]{#l00052}[ [52](structbght_1_1iht.html){.line}]{.lineno}[struct
]{.keyword}[iht](structbght_1_1iht.html){.code .hl_struct} {
:::

::: line
[]{#l00053}[ 53]{.lineno} [static_assert]{.keyword}(Threshold \< B,
[\"Threshold must be less than the bucket size\"]{.stringliteral});
:::

::: line
[]{#l00054}[ 54]{.lineno}
:::

::: line
[]{#l00055}[ 55]{.lineno} [using]{.keyword} value_type = pair\<Key, T\>;
:::

::: line
[]{#l00056}[ 56]{.lineno} [using]{.keyword} key_type = Key;
:::

::: line
[]{#l00057}[ 57]{.lineno} [using]{.keyword} mapped_type = T;
:::

::: line
[]{#l00058}[ 58]{.lineno} [using]{.keyword} atomic_pair_type =
cuda::atomic\<value_type, Scope\>;
:::

::: line
[]{#l00059}[ 59]{.lineno} [using]{.keyword} allocator_type = Allocator;
:::

::: line
[]{#l00060}[ 60]{.lineno} [using]{.keyword} hasher = Hash;
:::

::: line
[]{#l00061}[ 61]{.lineno} [using]{.keyword} size_type = std::size_t;
:::

::: line
[]{#l00062}[ 62]{.lineno}
:::

::: line
[]{#l00063}[ 63]{.lineno} [using]{.keyword} atomic_pair_allocator_type =
:::

::: line
[]{#l00064}[ 64]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<atomic_pair_type\>;
:::

::: line
[]{#l00065}[ 65]{.lineno} [using]{.keyword} pool_allocator_type =
:::

::: line
[]{#l00066}[ 66]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<[bool]{.keywordtype}\>;
:::

::: line
[]{#l00067}[ 67]{.lineno} [using]{.keyword} size_type_allocator_type =
:::

::: line
[]{#l00068}[ 68]{.lineno} [typename]{.keyword}
std::allocator_traits\<Allocator\>::rebind_alloc\<size_type\>;
:::

::: line
[]{#l00069}[ 69]{.lineno}
:::

::: line
[]{#l00070}[ 70]{.lineno} [static]{.keyword} [constexpr]{.keyword}
[auto]{.keyword} bucket_size = B;
:::

::: line
[]{#l00071}[ 71]{.lineno} [using]{.keyword} key_equal = KeyEqual;
:::

::: line
[]{#l00072}[ 72]{.lineno}
:::

::: line
[]{#l00083}[
[83](structbght_1_1iht.html#a02e2c454d005b7cf91363feb1477c7b9){.line}]{.lineno}
[iht](structbght_1_1iht.html#a02e2c454d005b7cf91363feb1477c7b9){.code
.hl_function}(std::size_t capacity,
:::

::: line
[]{#l00084}[ 84]{.lineno} Key sentinel_key,
:::

::: line
[]{#l00085}[ 85]{.lineno} T sentinel_value,
:::

::: line
[]{#l00086}[ 86]{.lineno} Allocator [const]{.keyword}& allocator =
Allocator{});
:::

::: line
[]{#l00090}[
[90](structbght_1_1iht.html#a725ba654f985af3bdbb95830f8e26776){.line}]{.lineno}
[iht](structbght_1_1iht.html#a725ba654f985af3bdbb95830f8e26776){.code
.hl_function}([const]{.keyword} [iht](structbght_1_1iht.html){.code
.hl_struct}& other);
:::

::: line
[]{#l00094}[
[94](structbght_1_1iht.html#a46dd032710006cc6cffcdc62a8da6564){.line}]{.lineno}
[iht](structbght_1_1iht.html#a46dd032710006cc6cffcdc62a8da6564){.code
.hl_function}([iht](structbght_1_1iht.html){.code .hl_struct}&&) =
[delete]{.keyword};
:::

::: line
[]{#l00098}[
[98](structbght_1_1iht.html#a9299d3e97a061b2319ccbfcc4d36fe71){.line}]{.lineno}
[iht](structbght_1_1iht.html){.code .hl_struct}&
[operator=](structbght_1_1iht.html#a9299d3e97a061b2319ccbfcc4d36fe71){.code
.hl_function}([const]{.keyword} [iht](structbght_1_1iht.html){.code
.hl_struct}&) = [delete]{.keyword};
:::

::: line
[]{#l00102}[
[102](structbght_1_1iht.html#a93b6e68ec5e324e1aae55cbbb900e24e){.line}]{.lineno}
[iht](structbght_1_1iht.html){.code .hl_struct}&
[operator=](structbght_1_1iht.html#a93b6e68ec5e324e1aae55cbbb900e24e){.code
.hl_function}([iht](structbght_1_1iht.html){.code .hl_struct}&&) =
[delete]{.keyword};
:::

::: line
[]{#l00106}[
[106](structbght_1_1iht.html#a1eb1c14a9683be082405b20cd4ac0fb5){.line}]{.lineno}
[\~iht](structbght_1_1iht.html#a1eb1c14a9683be082405b20cd4ac0fb5){.code
.hl_function}();
:::

::: line
[]{#l00110}[
[110](structbght_1_1iht.html#a076471ca81f32b45a0d91ac5a004946e){.line}]{.lineno}
[void]{.keywordtype}
[clear](structbght_1_1iht.html#a076471ca81f32b45a0d91ac5a004946e){.code
.hl_function}();
:::

::: line
[]{#l00111}[ 111]{.lineno}
:::

::: line
[]{#l00122}[ 122]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt\>
:::

::: line
[]{#l00123}[
[123](structbght_1_1iht.html#afcb16a7c2cc72a322d90e0bd8805cb50){.line}]{.lineno}
[bool]{.keywordtype}
[insert](structbght_1_1iht.html#afcb16a7c2cc72a322d90e0bd8805cb50){.code
.hl_function}(InputIt first, InputIt last, cudaStream_t stream = 0);
:::

::: line
[]{#l00124}[ 124]{.lineno}
:::

::: line
[]{#l00136}[ 136]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt\>
:::

::: line
[]{#l00137}[
[137](structbght_1_1iht.html#ac6aa71eb51d1f76879cd0ec3c398fadc){.line}]{.lineno}
[void]{.keywordtype}
[find](structbght_1_1iht.html#ac6aa71eb51d1f76879cd0ec3c398fadc){.code
.hl_function}(InputIt first, InputIt last, OutputIt output_begin,
cudaStream_t stream = 0);
:::

::: line
[]{#l00138}[ 138]{.lineno}
:::

::: line
[]{#l00151}[ 151]{.lineno} [template]{.keyword} \<[typename]{.keyword}
tile_type\>
:::

::: line
[]{#l00152}[
[152](structbght_1_1iht.html#abade7d2b541a2718bbae41be40a0ae40){.line}]{.lineno}
\_\_device\_\_ [bool]{.keywordtype}
[insert](structbght_1_1iht.html#abade7d2b541a2718bbae41be40a0ae40){.code
.hl_function}(value_type [const]{.keyword}& pair, tile_type
[const]{.keyword}& tile);
:::

::: line
[]{#l00153}[ 153]{.lineno}
:::

::: line
[]{#l00166}[ 166]{.lineno} [template]{.keyword} \<[typename]{.keyword}
tile_type\>
:::

::: line
[]{#l00167}[
[167](structbght_1_1iht.html#a5f0c222a207ee2050f6910a33a312b5d){.line}]{.lineno}
\_\_device\_\_ mapped_type
[find](structbght_1_1iht.html#a5f0c222a207ee2050f6910a33a312b5d){.code
.hl_function}(key_type [const]{.keyword}& key, tile_type
[const]{.keyword}& tile);
:::

::: line
[]{#l00168}[ 168]{.lineno}
:::

::: line
[]{#l00176}[ 176]{.lineno} [template]{.keyword} \<[typename]{.keyword}
RNG\>
:::

::: line
[]{#l00177}[
[177](structbght_1_1iht.html#acf9f5c49d7306e3567482b5a26b3b88b){.line}]{.lineno}
[void]{.keywordtype}
[randomize_hash_functions](structbght_1_1iht.html#acf9f5c49d7306e3567482b5a26b3b88b){.code
.hl_function}(RNG& rng);
:::

::: line
[]{#l00178}[ 178]{.lineno}
:::

::: line
[]{#l00183}[
[183](structbght_1_1iht.html#a0efa940784530baf1893c7cfbf901651){.line}]{.lineno}
size_type
[size](structbght_1_1iht.html#a0efa940784530baf1893c7cfbf901651){.code
.hl_function}(cudaStream_t stream = 0);
:::

::: line
[]{#l00184}[ 184]{.lineno}
:::

::: line
[]{#l00185}[ 185]{.lineno} [private]{.keyword}:
:::

::: line
[]{#l00186}[ 186]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00187}[ 187]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::tiled_insert_kernel(InputIt,
InputIt, HashMap);
:::

::: line
[]{#l00188}[ 188]{.lineno}
:::

::: line
[]{#l00189}[ 189]{.lineno} [template]{.keyword} \<[typename]{.keyword}
InputIt, [typename]{.keyword} OutputIt, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00190}[ 190]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::tiled_find_kernel(InputIt,
:::

::: line
[]{#l00191}[ 191]{.lineno} InputIt,
:::

::: line
[]{#l00192}[ 192]{.lineno} OutputIt,
:::

::: line
[]{#l00193}[ 193]{.lineno} HashMap);
:::

::: line
[]{#l00194}[ 194]{.lineno}
:::

::: line
[]{#l00195}[ 195]{.lineno} [template]{.keyword} \<[int]{.keywordtype}
BlockSize, [typename]{.keyword} InputT, [typename]{.keyword} HashMap\>
:::

::: line
[]{#l00196}[ 196]{.lineno} [friend]{.keyword} \_\_global\_\_
[void]{.keywordtype} detail::kernels::count_kernel([const]{.keyword}
InputT,
:::

::: line
[]{#l00197}[ 197]{.lineno} std::size_t\*,
:::

::: line
[]{#l00198}[ 198]{.lineno} HashMap);
:::

::: line
[]{#l00199}[ 199]{.lineno}
:::

::: line
[]{#l00200}[ 200]{.lineno} [static]{.keyword} [constexpr]{.keyword}
[auto]{.keyword} threshold\_ = Threshold;
:::

::: line
[]{#l00201}[ 201]{.lineno}
:::

::: line
[]{#l00202}[ 202]{.lineno} std::size_t capacity\_;
:::

::: line
[]{#l00203}[ 203]{.lineno} key_type sentinel_key\_{};
:::

::: line
[]{#l00204}[ 204]{.lineno} mapped_type sentinel_value\_{};
:::

::: line
[]{#l00205}[ 205]{.lineno} allocator_type allocator\_;
:::

::: line
[]{#l00206}[ 206]{.lineno} atomic_pair_allocator_type
atomic_pairs_allocator\_;
:::

::: line
[]{#l00207}[ 207]{.lineno} pool_allocator_type pool_allocator\_;
:::

::: line
[]{#l00208}[ 208]{.lineno} size_type_allocator_type
size_type_allocator\_;
:::

::: line
[]{#l00209}[ 209]{.lineno}
:::

::: line
[]{#l00210}[ 210]{.lineno} atomic_pair_type\* d_table\_{};
:::

::: line
[]{#l00211}[ 211]{.lineno} std::shared_ptr\<atomic_pair_type\> table\_;
:::

::: line
[]{#l00212}[ 212]{.lineno}
:::

::: line
[]{#l00213}[ 213]{.lineno} [bool]{.keywordtype}\* d_build_success\_;
:::

::: line
[]{#l00214}[ 214]{.lineno} std::shared_ptr\<bool\> build_success\_;
:::

::: line
[]{#l00215}[ 215]{.lineno}
:::

::: line
[]{#l00216}[ 216]{.lineno} Hash hfp\_;
:::

::: line
[]{#l00217}[ 217]{.lineno} Hash hf0\_;
:::

::: line
[]{#l00218}[ 218]{.lineno} Hash hf1\_;
:::

::: line
[]{#l00219}[ 219]{.lineno}
:::

::: line
[]{#l00220}[ 220]{.lineno} std::size_t num_buckets\_;
:::

::: line
[]{#l00221}[ 221]{.lineno}};
:::

::: line
[]{#l00222}[ 222]{.lineno}} [// namespace bght]{.comment}
:::

::: line
[]{#l00223}[ 223]{.lineno}
:::

::: line
[]{#l00224}[ 224]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T, [int]{.keywordtype} Threshold\>
:::

::: line
[]{#l00225}[ 225]{.lineno}[using]{.keyword} iht8 = [typename]{.keyword}
[bght::iht](structbght_1_1iht.html){.code .hl_struct}\<Key,
:::

::: line
[]{#l00226}[ 226]{.lineno} T,
:::

::: line
[]{#l00227}[ 227]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00228}[ 228]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00229}[ 229]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00230}[ 230]{.lineno}
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00231}[ 231]{.lineno} 8,
:::

::: line
[]{#l00232}[ 232]{.lineno} Threshold\>;
:::

::: line
[]{#l00233}[ 233]{.lineno}
:::

::: line
[]{#l00234}[ 234]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T, [int]{.keywordtype} Threshold = 12\>
:::

::: line
[]{#l00235}[ 235]{.lineno}[using]{.keyword} iht16 = [typename]{.keyword}
[bght::iht](structbght_1_1iht.html){.code .hl_struct}\<Key,
:::

::: line
[]{#l00236}[ 236]{.lineno} T,
:::

::: line
[]{#l00237}[ 237]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00238}[ 238]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00239}[ 239]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00240}[ 240]{.lineno}
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00241}[ 241]{.lineno} 16,
:::

::: line
[]{#l00242}[ 242]{.lineno} Threshold\>;
:::

::: line
[]{#l00243}[ 243]{.lineno}[template]{.keyword} \<[typename]{.keyword}
Key, [typename]{.keyword} T, [int]{.keywordtype} Threshold = 25\>
:::

::: line
[]{#l00244}[ 244]{.lineno}[using]{.keyword} iht32 = [typename]{.keyword}
[bght::iht](structbght_1_1iht.html){.code .hl_struct}\<Key,
:::

::: line
[]{#l00245}[ 245]{.lineno} T,
:::

::: line
[]{#l00246}[ 246]{.lineno} bght::universal_hash\<Key\>,
:::

::: line
[]{#l00247}[ 247]{.lineno} bght::equal_to\<Key\>,
:::

::: line
[]{#l00248}[ 248]{.lineno} cuda::thread_scope_device,
:::

::: line
[]{#l00249}[ 249]{.lineno}
[bght::cuda_allocator\<char\>](structbght_1_1cuda__allocator.html){.code
.hl_class},
:::

::: line
[]{#l00250}[ 250]{.lineno} 32,
:::

::: line
[]{#l00251}[ 251]{.lineno} Threshold\>;
:::

::: line
[]{#l00252}[ 252]{.lineno}
:::

::: line
[]{#l00253}[ 253]{.lineno}[#include
\<detail/iht_impl.cuh\>]{.preprocessor}
:::

::: {#astructbght_1_1cuda__allocator_html .ttc}
::: ttname
[bght::cuda_allocator\< char \>](structbght_1_1cuda__allocator.html)
:::
:::

::: {#astructbght_1_1iht_html .ttc}
::: ttname
[bght::iht](structbght_1_1iht.html)
:::

::: ttdoc
IHT IHT (iceberg hash table) is an associative static GPU hash table
that contains key-value pairs wi\...
:::

::: ttdef
**Definition:** iht.hpp:52
:::
:::

::: {#astructbght_1_1iht_html_a02e2c454d005b7cf91363feb1477c7b9 .ttc}
::: ttname
[bght::iht::iht](structbght_1_1iht.html#a02e2c454d005b7cf91363feb1477c7b9)
:::

::: ttdeci
iht(std::size_t capacity, Key sentinel_key, T sentinel_value, Allocator
const &allocator=Allocator{})
:::

::: ttdoc
Constructs the hash table with the specified capacity and uses the
specified sentinel key and value t\...
:::
:::

::: {#astructbght_1_1iht_html_a076471ca81f32b45a0d91ac5a004946e .ttc}
::: ttname
[bght::iht::clear](structbght_1_1iht.html#a076471ca81f32b45a0d91ac5a004946e)
:::

::: ttdeci
void clear()
:::

::: ttdoc
Clears the hash map and resets all slots.
:::
:::

::: {#astructbght_1_1iht_html_a0efa940784530baf1893c7cfbf901651 .ttc}
::: ttname
[bght::iht::size](structbght_1_1iht.html#a0efa940784530baf1893c7cfbf901651)
:::

::: ttdeci
size_type size(cudaStream_t stream=0)
:::

::: ttdoc
Compute the number of elements in the map.
:::
:::

::: {#astructbght_1_1iht_html_a1eb1c14a9683be082405b20cd4ac0fb5 .ttc}
::: ttname
[bght::iht::\~iht](structbght_1_1iht.html#a1eb1c14a9683be082405b20cd4ac0fb5)
:::

::: ttdeci
\~iht()
:::

::: ttdoc
Destructor that destroys the hash map and deallocate memory if no copies
exist.
:::
:::

::: {#astructbght_1_1iht_html_a46dd032710006cc6cffcdc62a8da6564 .ttc}
::: ttname
[bght::iht::iht](structbght_1_1iht.html#a46dd032710006cc6cffcdc62a8da6564)
:::

::: ttdeci
iht(iht &&)=delete
:::

::: ttdoc
Move constructor is currently deleted.
:::
:::

::: {#astructbght_1_1iht_html_a5f0c222a207ee2050f6910a33a312b5d .ttc}
::: ttname
[bght::iht::find](structbght_1_1iht.html#a5f0c222a207ee2050f6910a33a312b5d)
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

::: {#astructbght_1_1iht_html_a725ba654f985af3bdbb95830f8e26776 .ttc}
::: ttname
[bght::iht::iht](structbght_1_1iht.html#a725ba654f985af3bdbb95830f8e26776)
:::

::: ttdeci
iht(const iht &other)
:::

::: ttdoc
A shallow-copy constructor.
:::
:::

::: {#astructbght_1_1iht_html_a9299d3e97a061b2319ccbfcc4d36fe71 .ttc}
::: ttname
[bght::iht::operator=](structbght_1_1iht.html#a9299d3e97a061b2319ccbfcc4d36fe71)
:::

::: ttdeci
iht & operator=(const iht &)=delete
:::

::: ttdoc
The assignment operator is currently deleted.
:::
:::

::: {#astructbght_1_1iht_html_a93b6e68ec5e324e1aae55cbbb900e24e .ttc}
::: ttname
[bght::iht::operator=](structbght_1_1iht.html#a93b6e68ec5e324e1aae55cbbb900e24e)
:::

::: ttdeci
iht & operator=(iht &&)=delete
:::

::: ttdoc
The move assignment operator is currently deleted.
:::
:::

::: {#astructbght_1_1iht_html_abade7d2b541a2718bbae41be40a0ae40 .ttc}
::: ttname
[bght::iht::insert](structbght_1_1iht.html#abade7d2b541a2718bbae41be40a0ae40)
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

::: {#astructbght_1_1iht_html_ac6aa71eb51d1f76879cd0ec3c398fadc .ttc}
::: ttname
[bght::iht::find](structbght_1_1iht.html#ac6aa71eb51d1f76879cd0ec3c398fadc)
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

::: {#astructbght_1_1iht_html_acf9f5c49d7306e3567482b5a26b3b88b .ttc}
::: ttname
[bght::iht::randomize_hash_functions](structbght_1_1iht.html#acf9f5c49d7306e3567482b5a26b3b88b)
:::

::: ttdeci
void randomize_hash_functions(RNG &rng)
:::

::: ttdoc
Host-side API to randomize the hash functions used for the probing
scheme. This can be used when the \...
:::
:::

::: {#astructbght_1_1iht_html_afcb16a7c2cc72a322d90e0bd8805cb50 .ttc}
::: ttname
[bght::iht::insert](structbght_1_1iht.html#afcb16a7c2cc72a322d90e0bd8805cb50)
:::

::: ttdeci
bool insert(InputIt first, InputIt last, cudaStream_t stream=0)
:::

::: ttdoc
Host-side API for inserting all pairs defined by the input argument
iterators. All keys in the range \...
:::
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
