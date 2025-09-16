#pragma once
#include <string>
#include <vector>

struct TrainSet {
  // Images as floats in [0,1], row-major, size = N * 784
  std::vector<float> images;
  // Labels in {0..9}, size = N
  std::vector<int> labels;
  int N = 0;
};

struct TestSet {
  std::vector<float> images; // size = N * 784
  int N = 0;
};

// Load Kaggle CSVs (robust to header). Normalizes pixels to [0,1].
TrainSet load_train_csv(const std::string& path);
TestSet  load_test_csv (const std::string& path);
