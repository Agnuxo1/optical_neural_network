#include "training.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <cassert>

// Bilinear shift sampler (used for centering and augmentation)
static inline void bilinear_shift_28x28(const float* img, float* out, float dx, float dy) {
  const int H = 28, W = 28;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float sx = x - dx;
      float sy = y - dy;
      int x0 = (int)floorf(sx);
      int y0 = (int)floorf(sy);
      float ax = sx - x0;
      float ay = sy - y0;
      auto pix = [&](int yy, int xx) -> float {
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) return 0.f;
        return img[yy*W + xx];
      };
      float v00 = pix(y0,   x0);
      float v01 = pix(y0,   x0+1);
      float v10 = pix(y0+1, x0);
      float v11 = pix(y0+1, x0+1);
      float vx0 = v00*(1.f-ax) + v01*ax;
      float vx1 = v10*(1.f-ax) + v11*ax;
      out[y*W + x] = vx0*(1.f-ay) + vx1*ay;
    }
  }
}

// Center image by its intensity centroid
static inline void center_mass_28x28(const float* img, float* out) {
  const int H = 28, W = 28;
  double sum = 0.0, mx = 0.0, my = 0.0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float v = img[y*W + x];
      sum += v; mx += v * x; my += v * y;
    }
  }
  if (sum <= 1e-6) { std::memcpy(out, img, sizeof(float)*H*W); return; }
  float cx = float(mx / sum);
  float cy = float(my / sum);
  float tx = (W-1)*0.5f - cx;
  float ty = (H-1)*0.5f - cy;
  bilinear_shift_28x28(img, out, -tx, -ty);
}

// Deskew by shear using image moments (MNIST)
static inline void deskew_28x28(const float* img, float* out) {
  const int H = 28, W = 28;
  double sum = 0.0, mx = 0.0, my = 0.0;
  for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { float v = img[y*W + x]; sum += v; mx += v*x; my += v*y; }
  if (sum <= 1e-6) { std::memcpy(out, img, sizeof(float)*H*W); return; }
  double cx = mx / sum, cy = my / sum;
  double mu11 = 0.0, mu02 = 0.0;
  for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
    double dx = x - cx, dy = y - cy; double v = img[y*W + x]; mu11 += dx*dy*v; mu02 += dy*dy*v;
  }
  double alpha = (fabs(mu02) < 1e-6) ? 0.0 : (mu11 / mu02);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double xs = x - alpha * (y - cy);
      int x0 = (int)floor(xs);
      double ax = xs - x0;
      auto pix = [&](int xx)->float { if (xx < 0 || xx >= W) return 0.f; return img[y*W + xx]; };
      float v0 = pix(x0), v1 = pix(x0+1);
      out[y*W + x] = (float)(v0*(1.0-ax) + v1*ax);
    }
  }
}
  // Light data augmentation: rotation (±8 deg) + subpixel shift (±2 px)
  static inline void rotate_shift_28x28(const float* img, float* out, float angle_rad, float dx, float dy) {
    const int H = 28, W = 28;
    float cx = (W-1)*0.5f, cy = (H-1)*0.5f;
    float ca = cosf(angle_rad), sa = sinf(angle_rad);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
      float xr = x - cx;
      float yr = y - cy;
      float xr2 =  ca * xr + sa * yr;
      float yr2 = -sa * xr + ca * yr;
      float sx = xr2 + cx - dx;
      float sy = yr2 + cy - dy;
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        float ax = sx - x0;
        float ay = sy - y0;
        auto pix = [&](int yy, int xx) -> float {
          if (xx < 0 || xx >= W || yy < 0 || yy >= H) return 0.f;
          return img[yy*W + xx];
        };
        float v00 = pix(y0,   x0);
        float v01 = pix(y0,   x0+1);
        float v10 = pix(y0+1, x0);
        float v11 = pix(y0+1, x0+1);
        float vx0 = v00*(1.f-ax) + v01*ax;
        float vx1 = v10*(1.f-ax) + v11*ax;
        out[y*W + x] = vx0*(1.f-ay) + vx1*ay;
      }
    }
  }

static inline float accuracy_on_batch(const std::vector<float>& logits, const std::vector<int>& labels, int B) {
  int C = NUM_CLASSES;
  int correct = 0;
  for (int b = 0; b < B; ++b) {
    int best = 0; float bestv = logits[b*C + 0];
    for (int c = 1; c < C; ++c) {
      float v = logits[b*C + c];
      if (v > bestv) { bestv = v; best = c; }
    }
    if (best == labels[b]) correct++;
  }
  return float(correct) / float(B);
}

void train_model(const TrainSet& train,
                 OpticalParams& params,
                 int epochs, int batch, float lr) {
  const int N = train.N;
  const int S = IMG_SIZE;
  std::vector<int> indices(N);
  std::iota(indices.begin(), indices.end(), 0);

  DeviceBuffers dbuf{};
  allocate_device_buffers(dbuf, batch);
  FFTPlan fft{};
  create_fft_plan(fft, batch);

  std::vector<float> h_batch_in(batch * S);
  std::vector<int>   h_batch_lbl(batch);

  int iter = 0;
  const int steps_per_epoch = (N + batch - 1) / batch;
  const int total_steps = steps_per_epoch * epochs;
  const int warmup_steps = std::min(3 * steps_per_epoch, std::max(1, total_steps / 20));
  for (int ep = 1; ep <= epochs; ++ep) {
    // Shuffle each epoch
    std::mt19937 rng(1337u + ep);
    std::shuffle(indices.begin(), indices.end(), rng);

    float epoch_loss = 0.f;
    int   seen = 0;

    for (int start = 0; start < N; start += batch) {
      int B = std::min(batch, N - start);
      // pack batch with centering + augmentation (rotation ±8°, shift ±2 px)
      std::mt19937 rng_aug(1234u + ep*10007u + start);
      std::uniform_real_distribution<float> Ushift(-2.f, 2.f);
      std::uniform_real_distribution<float> Uang(-8.f, 8.f);
      for (int i = 0; i < B; ++i) {
        int idx = indices[start + i];
        float tmp[784], centered[784];
        deskew_28x28(&train.images[idx*784], tmp);
        center_mass_28x28(tmp, centered);
        float dx = Ushift(rng_aug);
        float dy = Ushift(rng_aug);
        float ang = Uang(rng_aug) * 3.14159265358979323846f / 180.f;
        rotate_shift_28x28(centered, &h_batch_in[i*S], ang, dx, dy);
        h_batch_lbl[i] = train.labels[idx];
      }
      // if B < batch, pad with zeros to satisfy cuFFT batch plan
      if (B < batch) {
        std::fill(h_batch_in.begin() + B*S, h_batch_in.end(), 0.f);
        for (int i = B; i < batch; ++i) h_batch_lbl[i] = 0;
      }

      iter++;
      // Cosine LR with warmup
      float lr_step = lr;
      int s = iter;
      if (s <= warmup_steps) {
        lr_step = lr * (float(s) / float(std::max(1, warmup_steps)));
      } else {
        float progress = float(s - warmup_steps) / float(std::max(1, total_steps - warmup_steps));
        float cosv = 0.5f * (1.f + cosf(3.14159265358979323846f * progress));
        lr_step = lr * cosv;
      }

      float loss = train_batch(h_batch_in.data(), h_batch_lbl.data(), batch,
                               params, dbuf, fft, lr_step, iter, true /*use_stage2*/);
      epoch_loss += loss * B;
      seen += B;
    }

    epoch_loss /= float(seen);
    std::cout << "[Epoch " << ep << "/" << epochs << "] "
              << "loss=" << epoch_loss << "\n";
  }

  destroy_fft_plan(fft);
  free_device_buffers(dbuf);
}

std::vector<int> run_inference(const TestSet& test,
                               const OpticalParams& params,
                               int batch) {
  const int N = test.N;
  const int S = IMG_SIZE;

  DeviceBuffers dbuf{};
  allocate_device_buffers(dbuf, batch);
  FFTPlan fft{};
  create_fft_plan(fft, batch);

  std::vector<float> h_batch_in(batch * S);
  std::vector<int> out_labels;
  out_labels.reserve(N);

  for (int start = 0; start < N; start += batch) {
    int B = std::min(batch, N - start);
    for (int i = 0; i < B; ++i) {
      float tmp[784];
      deskew_28x28(&test.images[(start+i)*784], tmp);
      center_mass_28x28(tmp, &h_batch_in[i*S]);
    }
    if (B < batch) {
      std::fill(h_batch_in.begin() + B*S, h_batch_in.end(), 0.f);
    }

    // Strong TTA: average logits over angles and small shifts
    const float degs[5] = { -8.f, -4.f, 0.f, 4.f, 8.f };
    const int shifts[5][2] = { {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1} };
    std::vector<float> aug(batch * S);
    std::vector<float> acc_logits(B*NUM_CLASSES, 0.f);
    for (int ai = 0; ai < 5; ++ai) {
      float ang = degs[ai] * 3.14159265358979323846f / 180.f;
      for (int si = 0; si < 5; ++si) {
        float dx = (float)shifts[si][0];
        float dy = (float)shifts[si][1];
        for (int i = 0; i < B; ++i) rotate_shift_28x28(&h_batch_in[i*S], &aug[i*S], ang, dx, dy);
        if (B < batch) std::fill(aug.begin() + B*S, aug.end(), 0.f);
        std::vector<float> logits(B*NUM_CLASSES);
        infer_batch_logits(aug.data(), batch, params, dbuf, fft, logits, true /*stage2*/, true /*EMA*/);
        for (int i = 0; i < B*NUM_CLASSES; ++i) acc_logits[i] += logits[i];
      }
    }
    for (int b = 0; b < B; ++b) {
      int best = 0; float bestv = acc_logits[b*NUM_CLASSES + 0];
      for (int c = 1; c < NUM_CLASSES; ++c) {
        float v = acc_logits[b*NUM_CLASSES + c];
        if (v > bestv) { bestv = v; best = c; }
      }
      out_labels.push_back(best);
    }
  }

  destroy_fft_plan(fft);
  free_device_buffers(dbuf);
  return out_labels;
}
