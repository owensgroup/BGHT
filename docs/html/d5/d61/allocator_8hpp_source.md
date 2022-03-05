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
-   [detail](../../dir_1ce12a969299dbe29f3c2a77345014ad.html){.el}
:::
:::

::: header
::: headertitle
::: title
allocator.hpp
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
[]{#l00018}[ 18]{.lineno}[#include \<cuda_runtime.h\>]{.preprocessor}
:::

::: line
[]{#l00019}[ 19]{.lineno}[#include
\<detail/cuda_helpers.cuh\>]{.preprocessor}
:::

::: line
[]{#l00020}[ 20]{.lineno}[namespace ]{.keyword}bght {
:::

::: line
[]{#l00021}[ 21]{.lineno}[template]{.keyword} \<[typename]{.keyword} T\>
:::

::: line
[]{#l00022}[
[22](../../dc/dee/structbght_1_1cuda__deleter.html){.line}]{.lineno}[struct
]{.keyword}[cuda_deleter](../../dc/dee/structbght_1_1cuda__deleter.html){.code
.hl_struct} {
:::

::: line
[]{#l00023}[ 23]{.lineno} [void]{.keywordtype} operator()(T\* p) {
cuda_try(cudaFree(p)); }
:::

::: line
[]{#l00024}[ 24]{.lineno}};
:::

::: line
[]{#l00025}[ 25]{.lineno}
:::

::: line
[]{#l00026}[ 26]{.lineno}[template]{.keyword} \<[class]{.keyword} T\>
:::

::: line
[]{#l00027}[
[27](../../d1/df4/structbght_1_1cuda__allocator.html){.line}]{.lineno}[struct
]{.keyword}[cuda_allocator](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_struct} {
:::

::: line
[]{#l00028}[ 28]{.lineno} [typedef]{.keyword} std::size_t size_type;
:::

::: line
[]{#l00029}[ 29]{.lineno} [typedef]{.keyword} std::ptrdiff_t
difference_type;
:::

::: line
[]{#l00030}[ 30]{.lineno}
:::

::: line
[]{#l00031}[ 31]{.lineno} [typedef]{.keyword} T value_type;
:::

::: line
[]{#l00032}[ 32]{.lineno} [typedef]{.keyword} T\* pointer;
:::

::: line
[]{#l00033}[ 33]{.lineno} [typedef]{.keyword} [const]{.keyword} T\*
const_pointer;
:::

::: line
[]{#l00034}[ 34]{.lineno} [typedef]{.keyword} T& reference;
:::

::: line
[]{#l00035}[ 35]{.lineno} [typedef]{.keyword} [const]{.keyword} T&
const_reference;
:::

::: line
[]{#l00036}[ 36]{.lineno}
:::

::: line
[]{#l00037}[ 37]{.lineno} [template]{.keyword} \<[class]{.keyword} U\>
:::

::: line
[]{#l00038}[
[38](../../d5/dbc/structbght_1_1cuda__allocator_1_1rebind.html){.line}]{.lineno}
[struct
]{.keyword}[rebind](../../d5/dbc/structbght_1_1cuda__allocator_1_1rebind.html){.code
.hl_struct} {
:::

::: line
[]{#l00039}[ 39]{.lineno} [typedef]{.keyword}
[cuda_allocator\<U\>](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_struct}
[other](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_struct};
:::

::: line
[]{#l00040}[ 40]{.lineno} };
:::

::: line
[]{#l00041}[ 41]{.lineno}
[cuda_allocator](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_struct}() = [default]{.keywordflow};
:::

::: line
[]{#l00042}[ 42]{.lineno} [template]{.keyword} \<[class]{.keyword} U\>
:::

::: line
[]{#l00043}[ 43]{.lineno} [constexpr]{.keyword}
[cuda_allocator](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_struct}([const]{.keyword}
[cuda_allocator\<U\>](../../d1/df4/structbght_1_1cuda__allocator.html){.code
.hl_struct}&) [noexcept]{.keyword} {}
:::

::: line
[]{#l00044}[ 44]{.lineno} T\* allocate(std::size_t n) {
:::

::: line
[]{#l00045}[ 45]{.lineno} [void]{.keywordtype}\* p =
[nullptr]{.keyword};
:::

::: line
[]{#l00046}[ 46]{.lineno} cuda_try(cudaMalloc(&p, n \*
[sizeof]{.keyword}(T)));
:::

::: line
[]{#l00047}[ 47]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T\*[\>]{.keyword}(p);
:::

::: line
[]{#l00048}[ 48]{.lineno} }
:::

::: line
[]{#l00049}[ 49]{.lineno} [void]{.keywordtype} deallocate(T\* p,
std::size_t n) [noexcept]{.keyword} { cuda_try(cudaFree(p)); }
:::

::: line
[]{#l00050}[ 50]{.lineno}};
:::

::: line
[]{#l00051}[ 51]{.lineno}} [// namespace bght]{.comment}
:::

::: {#astructbght_1_1cuda__allocator_1_1rebind_html .ttc}
::: ttname
[bght::cuda_allocator::rebind](../../d5/dbc/structbght_1_1cuda__allocator_1_1rebind.html)
:::

::: ttdef
**Definition:** allocator.hpp:38
:::
:::

::: {#astructbght_1_1cuda__allocator_html .ttc}
::: ttname
[bght::cuda_allocator](../../d1/df4/structbght_1_1cuda__allocator.html)
:::

::: ttdef
**Definition:** allocator.hpp:27
:::
:::

::: {#astructbght_1_1cuda__deleter_html .ttc}
::: ttname
[bght::cuda_deleter](../../dc/dee/structbght_1_1cuda__deleter.html)
:::

::: ttdef
**Definition:** allocator.hpp:22
:::
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
