#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>

namespace utils {

void set_seed(unsigned seed) {
  std::srand(seed);
}

void check_cuda(cudaError_t st, const char* msg) {
  if (st != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(st) << "\n";
    std::exit(1);
  }
}

void write_submission_csv(const std::string& path,
                          const std::vector<int>& labels) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot open submission file: " + path);
  }
  ofs << "ImageId,Label\n";
  for (size_t i = 0; i < labels.size(); ++i) {
    ofs << (i + 1) << "," << labels[i] << "\n";
  }
  ofs.close();
  std::cout << "[INFO] Submission saved to: " << path
            << " (" << labels.size() << " rows)\n";
}

} // namespace utils
