/*
 *   Copyright 2021 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once
#include <algorithm>
#include <iostream>
#include <optional>
#include <string_view>
#include <typeinfo>
#include <vector>

std::string str_tolower(const std::string_view s) {
  std::string output(s.length(), ' ');
  std::transform(s.begin(), s.end(), output.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return output;
}

// Finds an argument value
// auto arguments = std::vector<std::string>(argv, argv + argc);
// Example:
// auto k = get_arg_value<T>(arguments, "-flag")
// auto arguments = std::vector<std::string>(argv, argv + argc);
template <typename T>
std::optional<T> get_arg_value(const std::vector<std::string>& arguments,
                               const char* flag) {
  uint32_t first_argument = 1;
  for (uint32_t i = first_argument; i < arguments.size(); i++) {
    std::string_view argument = std::string_view(arguments[i]);
    auto key_start = argument.find_first_not_of("-");
    auto value_start = argument.find("=");

    bool failed = argument.length() == 0;              // there is an argument
    failed |= key_start == std::string::npos;          // it has a -
    failed |= value_start == std::string::npos;        // it has an =
    failed |= key_start > 2;                           // - or -- at beginning
    failed |= (value_start - key_start) == 0;          // there is a key
    failed |= (argument.length() - value_start) == 1;  // = is not last

    if (failed) {
      std::cout << "Invalid argument: " << argument << " ignored.\n";
      std::cout << "Use: -flag=value " << std::endl;
      std::terminate();
    }

    std::string_view argument_name = argument.substr(key_start, value_start - key_start);
    value_start++;  // ignore the =
    std::string_view argument_value =
        argument.substr(value_start, argument.length() - key_start);

    if (argument_name == std::string_view(flag)) {
      if constexpr (std::is_same<T, float>::value) {
        return static_cast<T>(std::strtof(argument_value.data(), nullptr));
      } else if constexpr (std::is_same<T, double>::value) {
        return static_cast<T>(std::strtod(argument_value.data(), nullptr));
      } else if constexpr (std::is_same<T, int>::value) {
        return static_cast<T>(std::strtol(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, long long>::value) {
        return static_cast<T>(std::strtoll(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, uint32_t>::value) {
        return static_cast<T>(std::strtoul(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, uint64_t>::value) {
        return static_cast<T>(std::strtoull(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, std::string>::value) {
        return std::string(argument_value);
      } else if constexpr (std::is_same<T, bool>::value) {
        return str_tolower(argument_value) == "true";
      } else {
        std::cout << "Unknown type" << std::endl;
        std::terminate();
      }
    }
  }
  return {};
}
