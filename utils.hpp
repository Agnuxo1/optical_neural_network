#pragma once
#include <string>
#include <vector>
#include <random>
#include <cuda_runtime.h>

namespace utils {

struct TrainConfig {
  std::string train_csv = "data/train.csv";
  std::string test_csv  = "data/test.csv";
  std::string submission_csv = "submission.csv";
  int epochs = 100;
  int batch  = 512;
  float lr   = 3e-3f;
  unsigned seed = 1337u;
  bool shuffle = true;
};

void set_seed(unsigned seed);
void check_cuda(cudaError_t st, const char* msg);
void write_submission_csv(const std::string& path,
                          const std::vector<int>& labels);
} // namespace utils
