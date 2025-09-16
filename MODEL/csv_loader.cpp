#include "csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

static inline bool is_integer(const std::string& s) {
  if (s.empty()) return false;
  size_t i = 0;
  if (s[0] == '-' || s[0] == '+') i = 1;
  for (; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return false;
  return true;
}

TrainSet load_train_csv(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) throw std::runtime_error("Cannot open train CSV: " + path);

  TrainSet set;
  set.images.reserve(42000 * 784);
  set.labels.reserve(42000);

  std::string line;
  bool header_checked = false;

  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);

    if (!header_checked) {
      std::string first;
      std::getline(ss, first, ',');
      if (!is_integer(first)) { header_checked = true; continue; }
      // It was not a header → first is actually a label
      set.labels.push_back(std::stoi(first));
      for (int i = 0; i < 784; ++i) {
        std::string tok; std::getline(ss, tok, ',');
        set.images.push_back(std::stof(tok) / 255.f);
      }
      set.N++;
      header_checked = true;
      continue;
    }

    std::string tok;
    std::getline(ss, tok, ',');
    set.labels.push_back(std::stoi(tok));
    for (int i = 0; i < 784; ++i) {
      std::getline(ss, tok, ',');
      set.images.push_back(std::stof(tok) / 255.f);
    }
    set.N++;
  }
  return set;
}

TestSet load_test_csv(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) throw std::runtime_error("Cannot open test CSV: " + path);

  TestSet set;
  set.images.reserve(28000 * 784);

  std::string line;
  bool header_checked = false;

  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);

    if (!header_checked) {
      std::string first;
      std::getline(ss, first, ',');
      if (!is_integer(first)) { header_checked = true; continue; }
      // first pixel
      set.images.push_back(std::stof(first) / 255.f);
      for (int i = 1; i < 784; ++i) {
        std::string tok; std::getline(ss, tok, ',');
        set.images.push_back(std::stof(tok) / 255.f);
      }
      set.N++;
      header_checked = true;
      continue;
    }

    for (int i = 0; i < 784; ++i) {
      std::string tok; std::getline(ss, tok, ',');
      set.images.push_back(std::stof(tok) / 255.f);
    }
    set.N++;
  }
  return set;
}
