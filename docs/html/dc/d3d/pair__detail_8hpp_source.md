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
pair_detail.hpp
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
[]{#l00018}[ 18]{.lineno}[#include \<type_traits\>]{.preprocessor}
:::

::: line
[]{#l00019}[ 19]{.lineno}[namespace ]{.keyword}bght {
:::

::: line
[]{#l00020}[ 20]{.lineno}[namespace ]{.keyword}detail {
:::

::: line
[]{#l00021}[ 21]{.lineno}[template]{.keyword} \<[typename]{.keyword} T\>
:::

::: line
[]{#l00022}[ 22]{.lineno}[constexpr]{.keyword} std::size_t
next_alignment() {
:::

::: line
[]{#l00023}[ 23]{.lineno} [constexpr]{.keyword} std::size_t n =
[sizeof]{.keyword}(T);
:::

::: line
[]{#l00024}[ 24]{.lineno} [if]{.keywordflow} (n \<= 4)
:::

::: line
[]{#l00025}[ 25]{.lineno} [return]{.keywordflow} 4;
:::

::: line
[]{#l00026}[ 26]{.lineno} [if]{.keywordflow} (n \<= 8)
:::

::: line
[]{#l00027}[ 27]{.lineno} [return]{.keywordflow} 8;
:::

::: line
[]{#l00028}[ 28]{.lineno} [return]{.keywordflow} 16;
:::

::: line
[]{#l00029}[ 29]{.lineno}}
:::

::: line
[]{#l00030}[ 30]{.lineno}[constexpr]{.keyword} std::size_t
next_alignment(std::size_t n) {
:::

::: line
[]{#l00031}[ 31]{.lineno} [if]{.keywordflow} (n \<= 4)
:::

::: line
[]{#l00032}[ 32]{.lineno} [return]{.keywordflow} 4;
:::

::: line
[]{#l00033}[ 33]{.lineno} [if]{.keywordflow} (n \<= 8)
:::

::: line
[]{#l00034}[ 34]{.lineno} [return]{.keywordflow} 8;
:::

::: line
[]{#l00035}[ 35]{.lineno} [return]{.keywordflow} 16;
:::

::: line
[]{#l00036}[ 36]{.lineno}}
:::

::: line
[]{#l00037}[ 37]{.lineno}
:::

::: line
[]{#l00038}[ 38]{.lineno}[template]{.keyword} \<[typename]{.keyword} T1,
[typename]{.keyword} T2\>
:::

::: line
[]{#l00039}[ 39]{.lineno}[constexpr]{.keyword} std::size_t pair_size() {
:::

::: line
[]{#l00040}[ 40]{.lineno} [return]{.keywordflow}
[sizeof]{.keyword}(T1) + [sizeof]{.keyword}(T2);
:::

::: line
[]{#l00041}[ 41]{.lineno}}
:::

::: line
[]{#l00042}[ 42]{.lineno}
:::

::: line
[]{#l00043}[ 43]{.lineno}[template]{.keyword} \<[typename]{.keyword} T1,
[typename]{.keyword} T2\>
:::

::: line
[]{#l00044}[ 44]{.lineno}[constexpr]{.keyword} std::size_t
pair_alignment() {
:::

::: line
[]{#l00045}[ 45]{.lineno} [return]{.keywordflow}
next_alignment(pair_size\<T1, T2\>());
:::

::: line
[]{#l00046}[ 46]{.lineno}}
:::

::: line
[]{#l00047}[ 47]{.lineno}
:::

::: line
[]{#l00048}[ 48]{.lineno}[template]{.keyword} \<[typename]{.keyword} T1,
[typename]{.keyword} T2\>
:::

::: line
[]{#l00049}[ 49]{.lineno}[constexpr]{.keyword} std::size_t
padding_size() {
:::

::: line
[]{#l00050}[ 50]{.lineno} [constexpr]{.keyword} [auto]{.keyword} psz =
pair_size\<T1, T2\>();
:::

::: line
[]{#l00051}[ 51]{.lineno} [constexpr]{.keyword} [auto]{.keyword} apsz =
next_alignment(pair_size\<T1, T2\>());
:::

::: line
[]{#l00052}[ 52]{.lineno} [if]{.keywordflow} (psz \> apsz) {
:::

::: line
[]{#l00053}[ 53]{.lineno} [constexpr]{.keyword} [auto]{.keyword} nsz =
(1ull + (psz / apsz)) \* apsz;
:::

::: line
[]{#l00054}[ 54]{.lineno} [return]{.keywordflow} nsz - psz;
:::

::: line
[]{#l00055}[ 55]{.lineno} }
:::

::: line
[]{#l00056}[ 56]{.lineno} [return]{.keywordflow} apsz - psz;
:::

::: line
[]{#l00057}[ 57]{.lineno}}
:::

::: line
[]{#l00058}[ 58]{.lineno}} [// namespace detail]{.comment}
:::

::: line
[]{#l00059}[ 59]{.lineno}} [// namespace bght]{.comment}
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
