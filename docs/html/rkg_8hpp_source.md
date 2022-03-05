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
rkg.hpp
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
[]{#l00018}[ 18]{.lineno}[#include \<filesystem\>]{.preprocessor}
:::

::: line
[]{#l00019}[ 19]{.lineno}[#include \<fstream\>]{.preprocessor}
:::

::: line
[]{#l00020}[ 20]{.lineno}[#include \<random\>]{.preprocessor}
:::

::: line
[]{#l00021}[ 21]{.lineno}[#include \<string\>]{.preprocessor}
:::

::: line
[]{#l00022}[ 22]{.lineno}[#include \<unordered_set\>]{.preprocessor}
:::

::: line
[]{#l00023}[ 23]{.lineno}[namespace ]{.keyword}rkg {
:::

::: line
[]{#l00024}[ 24]{.lineno}[template]{.keyword} \<[typename]{.keyword}
key_type, [typename]{.keyword} value_type\>
:::

::: line
[]{#l00025}[ 25]{.lineno}value_type generate_value(key_type in) {
:::

::: line
[]{#l00026}[ 26]{.lineno} [return]{.keywordflow} in + 1;
:::

::: line
[]{#l00027}[ 27]{.lineno}}
:::

::: line
[]{#l00028}[ 28]{.lineno}
:::

::: line
[]{#l00029}[ 29]{.lineno}[template]{.keyword} \<[typename]{.keyword}
key_type, [typename]{.keyword} value_type, [typename]{.keyword}
[size_t]{.keywordtype}ype\>
:::

::: line
[]{#l00030}[ 30]{.lineno}[void]{.keywordtype}
generate_uniform_unique_pairs(std::vector\<key_type\>& keys,
:::

::: line
[]{#l00031}[ 31]{.lineno} std::vector\<value_type\>& values,
:::

::: line
[]{#l00032}[ 32]{.lineno} size_type num_keys,
:::

::: line
[]{#l00033}[ 33]{.lineno} [bool]{.keywordtype} cache =
[false]{.keyword},
:::

::: line
[]{#l00034}[ 34]{.lineno} key_type min_key = 0) {
:::

::: line
[]{#l00035}[ 35]{.lineno} keys.resize(num_keys);
:::

::: line
[]{#l00036}[ 36]{.lineno} values.resize(num_keys);
:::

::: line
[]{#l00037}[ 37]{.lineno} [unsigned]{.keywordtype} seed = 1;
:::

::: line
[]{#l00038}[ 38]{.lineno} [// bool cache = true;]{.comment}
:::

::: line
[]{#l00039}[ 39]{.lineno} std::string dataset_dir =
[\"dataset\"]{.stringliteral};
:::

::: line
[]{#l00040}[ 40]{.lineno} std::string dataset_name =
std::to_string(num_keys) + [\"\_\"]{.stringliteral} +
std::to_string(seed);
:::

::: line
[]{#l00041}[ 41]{.lineno} std::string dataset_path = dataset_dir +
[\"/\"]{.stringliteral} + dataset_name;
:::

::: line
[]{#l00042}[ 42]{.lineno} [if]{.keywordflow} (cache) {
:::

::: line
[]{#l00043}[ 43]{.lineno} [if]{.keywordflow}
(std::filesystem::exists(dataset_dir)) {
:::

::: line
[]{#l00044}[ 44]{.lineno} [if]{.keywordflow}
(std::filesystem::exists(dataset_path)) {
:::

::: line
[]{#l00045}[ 45]{.lineno} std::cout \<\< [\"Reading cached
keys..\"]{.stringliteral} \<\< std::endl;
:::

::: line
[]{#l00046}[ 46]{.lineno} std::ifstream dataset(dataset_path,
std::ios::binary);
:::

::: line
[]{#l00047}[ 47]{.lineno}
dataset.read(([char]{.keywordtype}\*)keys.data(),
[sizeof]{.keyword}(key_type) \* num_keys);
:::

::: line
[]{#l00048}[ 48]{.lineno}
dataset.read(([char]{.keywordtype}\*)values.data(),
[sizeof]{.keyword}(value_type) \* num_keys);
:::

::: line
[]{#l00049}[ 49]{.lineno} dataset.close();
:::

::: line
[]{#l00050}[ 50]{.lineno} [return]{.keywordflow};
:::

::: line
[]{#l00051}[ 51]{.lineno} }
:::

::: line
[]{#l00052}[ 52]{.lineno} } [else]{.keywordflow} {
:::

::: line
[]{#l00053}[ 53]{.lineno}
std::filesystem::create_directory(dataset_dir);
:::

::: line
[]{#l00054}[ 54]{.lineno} }
:::

::: line
[]{#l00055}[ 55]{.lineno} }
:::

::: line
[]{#l00056}[ 56]{.lineno} std::random_device rd;
:::

::: line
[]{#l00057}[ 57]{.lineno} std::mt19937 rng(seed);
:::

::: line
[]{#l00058}[ 58]{.lineno} [auto]{.keyword} max_key =
std::numeric_limits\<key_type\>::max() - 1;
:::

::: line
[]{#l00059}[ 59]{.lineno} std::uniform_int_distribution\<key_type\>
uni(min_key, max_key);
:::

::: line
[]{#l00060}[ 60]{.lineno} std::unordered_set\<key_type\> unique_keys;
:::

::: line
[]{#l00061}[ 61]{.lineno} [while]{.keywordflow} (unique_keys.size() \<
num_keys) {
:::

::: line
[]{#l00062}[ 62]{.lineno} unique_keys.insert(uni(rng));
:::

::: line
[]{#l00063}[ 63]{.lineno} [// unique_keys.insert(unique_keys.size() +
1);]{.comment}
:::

::: line
[]{#l00064}[ 64]{.lineno} }
:::

::: line
[]{#l00065}[ 65]{.lineno} std::copy(unique_keys.cbegin(),
unique_keys.cend(), keys.begin());
:::

::: line
[]{#l00066}[ 66]{.lineno} std::shuffle(keys.begin(), keys.end(), rng);
:::

::: line
[]{#l00067}[ 67]{.lineno}
:::

::: line
[]{#l00068}[ 68]{.lineno}[#ifdef \_WIN32]{.preprocessor}
:::

::: line
[]{#l00069}[ 69]{.lineno} [// OpenMP + windows don\'t allow unsigned
loops]{.comment}
:::

::: line
[]{#l00070}[ 70]{.lineno} [for]{.keywordflow} (uint32_t i = 0; i \<
unique_keys.size(); i++) {
:::

::: line
[]{#l00071}[ 71]{.lineno} values\[i\] = generate_value\<key_type,
value_type\>(keys\[i\]);
:::

::: line
[]{#l00072}[ 72]{.lineno} }
:::

::: line
[]{#l00073}[ 73]{.lineno}[#else]{.preprocessor}
:::

::: line
[]{#l00074}[ 74]{.lineno}
:::

::: line
[]{#l00075}[ 75]{.lineno} [for]{.keywordflow} (uint32_t i = 0; i \<
unique_keys.size(); i++) {
:::

::: line
[]{#l00076}[ 76]{.lineno} values\[i\] = generate_value\<key_type,
value_type\>(keys\[i\]);
:::

::: line
[]{#l00077}[ 77]{.lineno} }
:::

::: line
[]{#l00078}[ 78]{.lineno}[#endif]{.preprocessor}
:::

::: line
[]{#l00079}[ 79]{.lineno}
:::

::: line
[]{#l00080}[ 80]{.lineno} [if]{.keywordflow} (cache) {
:::

::: line
[]{#l00081}[ 81]{.lineno} std::cout \<\< [\"Caching..\"]{.stringliteral}
\<\< std::endl;
:::

::: line
[]{#l00082}[ 82]{.lineno} std::ofstream dataset(dataset_path,
std::ios::binary);
:::

::: line
[]{#l00083}[ 83]{.lineno}
dataset.write(([char]{.keywordtype}\*)keys.data(),
[sizeof]{.keyword}(key_type) \* num_keys);
:::

::: line
[]{#l00084}[ 84]{.lineno}
dataset.write(([char]{.keywordtype}\*)values.data(),
[sizeof]{.keyword}(value_type) \* num_keys);
:::

::: line
[]{#l00085}[ 85]{.lineno} dataset.close();
:::

::: line
[]{#l00086}[ 86]{.lineno} }
:::

::: line
[]{#l00087}[ 87]{.lineno}}
:::

::: line
[]{#l00088}[ 88]{.lineno}
:::

::: line
[]{#l00089}[ 89]{.lineno}[template]{.keyword} \<[typename]{.keyword}
key_type, [typename]{.keyword} [size_t]{.keywordtype}ype\>
:::

::: line
[]{#l00090}[ 90]{.lineno}[void]{.keywordtype}
generate_uniform_unique_keys(std::vector\<key_type\>& keys, size_type
num_keys) {
:::

::: line
[]{#l00091}[ 91]{.lineno} keys.resize(num_keys);
:::

::: line
[]{#l00092}[ 92]{.lineno} [unsigned]{.keywordtype} seed = 1;
:::

::: line
[]{#l00093}[ 93]{.lineno} std::random_device rd;
:::

::: line
[]{#l00094}[ 94]{.lineno} std::mt19937 rng(seed);
:::

::: line
[]{#l00095}[ 95]{.lineno} [auto]{.keyword} max_key =
std::numeric_limits\<key_type\>::max() - 1;
:::

::: line
[]{#l00096}[ 96]{.lineno} std::uniform_int_distribution\<key_type\>
uni(0, max_key);
:::

::: line
[]{#l00097}[ 97]{.lineno} std::unordered_set\<key_type\> unique_keys;
:::

::: line
[]{#l00098}[ 98]{.lineno} [while]{.keywordflow} (unique_keys.size() \<
num_keys) {
:::

::: line
[]{#l00099}[ 99]{.lineno} unique_keys.insert(uni(rng));
:::

::: line
[]{#l00100}[ 100]{.lineno} }
:::

::: line
[]{#l00101}[ 101]{.lineno} std::copy(unique_keys.cbegin(),
unique_keys.cend(), keys.begin());
:::

::: line
[]{#l00102}[ 102]{.lineno} std::shuffle(keys.begin(), keys.end(), rng);
:::

::: line
[]{#l00103}[ 103]{.lineno}}
:::

::: line
[]{#l00104}[ 104]{.lineno}} [// namespace rkg]{.comment}
:::
:::
:::

------------------------------------------------------------------------

[Generated by [![doxygen](doxygen.svg){.footer width="104"
height="31"}](https://www.doxygen.org/index.html) 1.9.2]{.small}
