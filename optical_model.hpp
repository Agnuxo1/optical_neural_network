#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

// Image geometry
constexpr int IMG_H = 28;
constexpr int IMG_W = 28;
constexpr int IMG_SIZE = IMG_H * IMG_W;
constexpr int NUM_CLASSES = 10;

// Model parameters (one optical block + linear head)
struct OpticalParams {
  // Learnable raw parameters (unconstrained):
  // amplitude A = softplus(a_raw) + eps
  // phase     P = pi * tanh(p_raw)
  std::vector<float> a_raw; // size IMG_SIZE
  std::vector<float> p_raw; // size IMG_SIZE
  // Second optical stage
  std::vector<float> a2_raw; // size IMG_SIZE
  std::vector<float> p2_raw; // size IMG_SIZE

  // Linear head: logits = W * y + b, where y is flattened optical feature (IMG_SIZE)
  // W shape: [NUM_CLASSES, IMG_SIZE], b shape: [NUM_CLASSES]
  std::vector<float> W;
  std::vector<float> b;

  // Adam moments
  std::vector<float> ma_a, va_a;
  std::vector<float> ma_p, va_p;
  std::vector<float> ma_a2, va_a2;
  std::vector<float> ma_p2, va_p2;
  std::vector<float> ma_W, va_W;
  std::vector<float> ma_b, va_b;

  // EMA weights for inference (optional)
  std::vector<float> ema_a, ema_p, ema_a2, ema_p2, ema_W, ema_b;
};

// Device buffers container (allocated once)
struct DeviceBuffers {
  // Inputs/labels
  float* d_batch_in = nullptr;     // [B, IMG_SIZE]
  int*   d_batch_lbl = nullptr;    // [B]

  // Optical fields / intermediates
  cufftComplex* d_field = nullptr; // [B, IMG_SIZE] complex input to FFT
  cufftComplex* d_freq  = nullptr; // [B, IMG_SIZE] complex after FFT
  cufftComplex* d_field2 = nullptr; // [B, IMG_SIZE] second stage field
  cufftComplex* d_freq2  = nullptr; // [B, IMG_SIZE] second stage freq

  float* d_I = nullptr;            // [B, IMG_SIZE] intensity |U|^2
  float* d_y = nullptr;            // [B, IMG_SIZE] nonlinearity log1p(I)
  float* d_I2 = nullptr;           // [B, IMG_SIZE] second stage intensity
  float* d_y2 = nullptr;           // [B, IMG_SIZE] second stage nonlinearity

  // Classifier
  float* d_logits = nullptr;       // [B, NUM_CLASSES]
  float* d_probs  = nullptr;       // [B, NUM_CLASSES]
  float* d_grad_logits = nullptr;  // [B, NUM_CLASSES]
  float* d_grad_y = nullptr;       // [B, IMG_SIZE]
  // For two-stage backprop
  cufftComplex* d_grad_freq2 = nullptr;   // [B, IMG_SIZE]
  cufftComplex* d_grad_field2 = nullptr;  // [B, IMG_SIZE]

  // Grad wrt FFT outputs
  cufftComplex* d_grad_freq = nullptr;   // [B, IMG_SIZE]
  cufftComplex* d_grad_field = nullptr;  // [B, IMG_SIZE]

  // Cached masks (elementwise) per iteration
  float* d_A = nullptr;            // [IMG_SIZE]
  float* d_P = nullptr;            // [IMG_SIZE]
  float* d_cosP = nullptr;         // [IMG_SIZE]
  float* d_sinP = nullptr;         // [IMG_SIZE]
  float* d_A2 = nullptr;           // [IMG_SIZE]
  float* d_P2 = nullptr;           // [IMG_SIZE]
  float* d_cosP2 = nullptr;        // [IMG_SIZE]
  float* d_sinP2 = nullptr;        // [IMG_SIZE]

  // Gradients accumulation (parameters)
  float* d_g_a_raw = nullptr;      // [IMG_SIZE]
  float* d_g_p_raw = nullptr;      // [IMG_SIZE]
  float* d_g_a2_raw = nullptr;     // [IMG_SIZE]
  float* d_g_p2_raw = nullptr;     // [IMG_SIZE]
  float* d_g_W = nullptr;          // [NUM_CLASSES * IMG_SIZE]
  float* d_g_b = nullptr;          // [NUM_CLASSES]
};

// cuFFT plan
struct FFTPlan {
  cufftHandle plan_fwd{};
  cufftHandle plan_inv{};
  int batch = 0;
};

// Allocate / free device buffers
void allocate_device_buffers(DeviceBuffers& dbuf, int batch);
void free_device_buffers(DeviceBuffers& dbuf);

// Initialize model params
void init_params(OpticalParams& p, unsigned seed);

// Single training step on a batch (forward+backward+update)
float train_batch(const float* h_batch_in, const int* h_batch_lbl,
                  int B,
                  OpticalParams& params,
                  DeviceBuffers& dbuf,
                  FFTPlan& fft,
                  float lr,
                  int t_adam,
                  bool use_stage2); // toggle second optical stage

// Inference for a batch (returns predicted labels)
void infer_batch(const float* h_batch_in, int B,
                 const OpticalParams& params,
                 DeviceBuffers& dbuf,
                 FFTPlan& fft,
                 std::vector<int>& out_labels,
                 bool use_stage2,
                 bool use_ema);

// Variant: return logits instead of labels (averaging-friendly)
void infer_batch_logits(const float* h_batch_in, int B,
                        const OpticalParams& params,
                        DeviceBuffers& dbuf,
                        FFTPlan& fft,
                        std::vector<float>& out_logits,
                        bool use_stage2,
                        bool use_ema);

// Create cuFFT plans for batch B (IMG_H x IMG_W, complex-to-complex)
void create_fft_plan(FFTPlan& fft, int B);
void destroy_fft_plan(FFTPlan& fft);
