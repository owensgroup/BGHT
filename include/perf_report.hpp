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

void std_cout_perf_report(float insertion_s,
                          float find_s,
                          std::size_t num_insertions,
                          std::size_t num_finds) {
  std::cout << "inserted: " << num_insertions << " keys" << '\n';
  std::cout << "finds: " << num_finds << " keys" << '\n';

  double insertion_rate = double(num_insertions) * 1e-6 / insertion_s;
  double find_rate = double(num_finds) * 1e-6 / find_s;

  std::cout << "insert_rate: " << insertion_rate << " Mkey/s" << '\n';
  std::cout << "find_rate: " << find_rate << " Mkey/s" << '\n';
}
