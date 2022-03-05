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
cmd.hpp
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
[]{#l00018}[ 18]{.lineno}[#include \<algorithm\>]{.preprocessor}
:::

::: line
[]{#l00019}[ 19]{.lineno}[#include \<iostream\>]{.preprocessor}
:::

::: line
[]{#l00020}[ 20]{.lineno}[#include \<optional\>]{.preprocessor}
:::

::: line
[]{#l00021}[ 21]{.lineno}[#include \<string_view\>]{.preprocessor}
:::

::: line
[]{#l00022}[ 22]{.lineno}[#include \<typeinfo\>]{.preprocessor}
:::

::: line
[]{#l00023}[ 23]{.lineno}[#include \<vector\>]{.preprocessor}
:::

::: line
[]{#l00024}[ 24]{.lineno}
:::

::: line
[]{#l00025}[ 25]{.lineno}std::string str_tolower([const]{.keyword}
std::string_view s) {
:::

::: line
[]{#l00026}[ 26]{.lineno} std::string output(s.length(), [\'
\']{.charliteral});
:::

::: line
[]{#l00027}[ 27]{.lineno} std::transform(s.begin(), s.end(),
output.begin(), \[\]([unsigned]{.keywordtype} [char]{.keywordtype} c) {
:::

::: line
[]{#l00028}[ 28]{.lineno} return std::tolower(c);
:::

::: line
[]{#l00029}[ 29]{.lineno} });
:::

::: line
[]{#l00030}[ 30]{.lineno} [return]{.keywordflow} output;
:::

::: line
[]{#l00031}[ 31]{.lineno}}
:::

::: line
[]{#l00032}[ 32]{.lineno}
:::

::: line
[]{#l00033}[ 33]{.lineno}[// Finds an argument value]{.comment}
:::

::: line
[]{#l00034}[ 34]{.lineno}[// auto arguments =
std::vector\<std::string\>(argv, argv + argc);]{.comment}
:::

::: line
[]{#l00035}[ 35]{.lineno}[// Example:]{.comment}
:::

::: line
[]{#l00036}[ 36]{.lineno}[// auto k = get_arg_value\<T\>(arguments,
\"-flag\")]{.comment}
:::

::: line
[]{#l00037}[ 37]{.lineno}[// auto arguments =
std::vector\<std::string\>(argv, argv + argc);]{.comment}
:::

::: line
[]{#l00038}[ 38]{.lineno}[template]{.keyword} \<[typename]{.keyword} T\>
:::

::: line
[]{#l00039}[ 39]{.lineno}std::optional\<T\>
get_arg_value([const]{.keyword} std::vector\<std::string\>& arguments,
:::

::: line
[]{#l00040}[ 40]{.lineno} [const]{.keyword} [char]{.keywordtype}\* flag)
{
:::

::: line
[]{#l00041}[ 41]{.lineno} uint32_t first_argument = 1;
:::

::: line
[]{#l00042}[ 42]{.lineno} [for]{.keywordflow} (uint32_t i =
first_argument; i \< arguments.size(); i++) {
:::

::: line
[]{#l00043}[ 43]{.lineno} std::string_view argument =
std::string_view(arguments\[i\]);
:::

::: line
[]{#l00044}[ 44]{.lineno} [auto]{.keyword} key_start =
argument.find_first_not_of([\"-\"]{.stringliteral});
:::

::: line
[]{#l00045}[ 45]{.lineno} [auto]{.keyword} value_start =
argument.find([\"=\"]{.stringliteral});
:::

::: line
[]{#l00046}[ 46]{.lineno}
:::

::: line
[]{#l00047}[ 47]{.lineno} [bool]{.keywordtype} failed =
argument.length() == 0; [// there is an argument]{.comment}
:::

::: line
[]{#l00048}[ 48]{.lineno} failed \|= key_start == std::string::npos; [//
it has a -]{.comment}
:::

::: line
[]{#l00049}[ 49]{.lineno} failed \|= value_start == std::string::npos;
[// it has an =]{.comment}
:::

::: line
[]{#l00050}[ 50]{.lineno} failed \|= key_start \> 2; [// - or \-- at
beginning]{.comment}
:::

::: line
[]{#l00051}[ 51]{.lineno} failed \|= (value_start - key_start) == 0; [//
there is a key]{.comment}
:::

::: line
[]{#l00052}[ 52]{.lineno} failed \|= (argument.length() - value_start)
== 1; [// = is not last]{.comment}
:::

::: line
[]{#l00053}[ 53]{.lineno}
:::

::: line
[]{#l00054}[ 54]{.lineno} [if]{.keywordflow} (failed) {
:::

::: line
[]{#l00055}[ 55]{.lineno} std::cout \<\< [\"Invalid argument:
\"]{.stringliteral} \<\< argument \<\< [\"
ignored.\\n\"]{.stringliteral};
:::

::: line
[]{#l00056}[ 56]{.lineno} std::cout \<\< [\"Use: -flag=value
\"]{.stringliteral} \<\< std::endl;
:::

::: line
[]{#l00057}[ 57]{.lineno} std::terminate();
:::

::: line
[]{#l00058}[ 58]{.lineno} }
:::

::: line
[]{#l00059}[ 59]{.lineno}
:::

::: line
[]{#l00060}[ 60]{.lineno} std::string_view argument_name =
argument.substr(key_start, value_start - key_start);
:::

::: line
[]{#l00061}[ 61]{.lineno} value_start++; [// ignore the =]{.comment}
:::

::: line
[]{#l00062}[ 62]{.lineno} std::string_view argument_value =
:::

::: line
[]{#l00063}[ 63]{.lineno} argument.substr(value_start,
argument.length() - key_start);
:::

::: line
[]{#l00064}[ 64]{.lineno}
:::

::: line
[]{#l00065}[ 65]{.lineno} [if]{.keywordflow} (argument_name ==
std::string_view(flag)) {
:::

::: line
[]{#l00066}[ 66]{.lineno} [if]{.keywordflow} [constexpr]{.keyword}
(std::is_same\<T, float\>::value) {
:::

::: line
[]{#l00067}[ 67]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T[\>]{.keyword}(std::strtof(argument_value.data(),
[nullptr]{.keyword}));
:::

::: line
[]{#l00068}[ 68]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, double\>::value) {
:::

::: line
[]{#l00069}[ 69]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T[\>]{.keyword}(std::strtod(argument_value.data(),
[nullptr]{.keyword}));
:::

::: line
[]{#l00070}[ 70]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, int\>::value) {
:::

::: line
[]{#l00071}[ 71]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T[\>]{.keyword}(std::strtol(argument_value.data(),
[nullptr]{.keyword}, 10));
:::

::: line
[]{#l00072}[ 72]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, long long\>::value) {
:::

::: line
[]{#l00073}[ 73]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T[\>]{.keyword}(std::strtoll(argument_value.data(),
[nullptr]{.keyword}, 10));
:::

::: line
[]{#l00074}[ 74]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, uint32_t\>::value) {
:::

::: line
[]{#l00075}[ 75]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T[\>]{.keyword}(std::strtoul(argument_value.data(),
[nullptr]{.keyword}, 10));
:::

::: line
[]{#l00076}[ 76]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, uint64_t\>::value) {
:::

::: line
[]{#l00077}[ 77]{.lineno} [return]{.keywordflow}
[static_cast\<]{.keyword}T[\>]{.keyword}(std::strtoull(argument_value.data(),
[nullptr]{.keyword}, 10));
:::

::: line
[]{#l00078}[ 78]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, std::string\>::value) {
:::

::: line
[]{#l00079}[ 79]{.lineno} [return]{.keywordflow}
std::string(argument_value);
:::

::: line
[]{#l00080}[ 80]{.lineno} } [else]{.keywordflow} [if]{.keywordflow}
[constexpr]{.keyword} (std::is_same\<T, bool\>::value) {
:::

::: line
[]{#l00081}[ 81]{.lineno} [return]{.keywordflow}
str_tolower(argument_value) == [\"true\"]{.stringliteral};
:::

::: line
[]{#l00082}[ 82]{.lineno} } [else]{.keywordflow} {
:::

::: line
[]{#l00083}[ 83]{.lineno} std::cout \<\< [\"Unknown
type\"]{.stringliteral} \<\< std::endl;
:::

::: line
[]{#l00084}[ 84]{.lineno} std::terminate();
:::

::: line
[]{#l00085}[ 85]{.lineno} }
:::

::: line
[]{#l00086}[ 86]{.lineno} }
:::

::: line
[]{#l00087}[ 87]{.lineno} }
:::

::: line
[]{#l00088}[ 88]{.lineno} [return]{.keywordflow} {};
:::

::: line
[]{#l00089}[ 89]{.lineno}}
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](../../doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
