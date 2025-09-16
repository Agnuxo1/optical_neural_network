#pragma once
#include "optical_model.hpp"
#include "csv_loader.hpp"
#include "utils.hpp"

struct Metrics {
  float loss = 0.f;
  float acc  = 0.f;
};

// Train for E epochs. Returns last epoch metrics.
void train_model(const TrainSet& train,
                 OpticalParams& params,
                 int epochs, int batch, float lr);

// Run inference on test set and produce labels
std::vector<int> run_inference(const TestSet& test,
                               const OpticalParams& params,
                               int batch);
