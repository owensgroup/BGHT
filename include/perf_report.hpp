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
