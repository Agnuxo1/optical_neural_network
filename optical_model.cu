#include "optical_model.hpp"
#include "utils.hpp"
#include <vector>
#include <random>
#include <cstring>
#include <iostream>

extern __global__ void k_build_masks(const float*, const float*, float*, float*, float*, float*, int);
extern __global__ void k_modulate(const float*, const float*, const float*, const float*, cufftComplex*, int);
extern __global__ void k_intensity_nl(const cufftComplex*, float*, float*, int);
extern __global__ void k_linear_forward(const float*, const float*, const float*, float*, int, int, int);
extern __global__ void k_softmax_xent(const float*, const int*, float*, float*, int, int, float*);
extern __global__ void k_linear_backward_to_y(const float*, const float*, float*, int, int, int);
extern __global__ void k_linear_accum_grads_Wb(const float*, const float*, float*, float*, int, int, int);
extern __global__ void k_backprop_y_to_I(const float*, const float*, float*, int);
extern __global__ void k_backprop_I_to_freq(const float*, const cufftComplex*, cufftComplex*, int);
extern __global__ void k_scale_complex(cufftComplex*, float, int);
extern __global__ void k_backprop_field_to_params_exact(const float*, const float*, const float*, const float*, const float*, const float*, const cufftComplex*, float*, float*, int, int);
extern __global__ void k_backprop_field_to_input(const cufftComplex*, const float*, const float*, const float*, float*, int, int);

// Small launch helper (local copy)
static inline dim3 make_grid(int N, int block) {
  return dim3((N + block - 1) / block);
}

void allocate_device_buffers(DeviceBuffers& d, int B) {
  using utils::check_cuda;
  size_t S = IMG_SIZE;

  check_cuda(cudaMalloc(&d.d_batch_in,  sizeof(float) * B * S), "alloc d_batch_in");
  check_cuda(cudaMalloc(&d.d_batch_lbl, sizeof(int)   * B),     "alloc d_batch_lbl");

  check_cuda(cudaMalloc(&d.d_field,      sizeof(cufftComplex) * B * S), "alloc d_field");
  check_cuda(cudaMalloc(&d.d_freq,       sizeof(cufftComplex) * B * S), "alloc d_freq");
  check_cuda(cudaMalloc(&d.d_field2,     sizeof(cufftComplex) * B * S), "alloc d_field2");
  check_cuda(cudaMalloc(&d.d_freq2,      sizeof(cufftComplex) * B * S), "alloc d_freq2");
  check_cuda(cudaMalloc(&d.d_I,          sizeof(float) * B * S),        "alloc d_I");
  check_cuda(cudaMalloc(&d.d_y,          sizeof(float) * B * S),        "alloc d_y");
  check_cuda(cudaMalloc(&d.d_I2,         sizeof(float) * B * S),        "alloc d_I2");
  check_cuda(cudaMalloc(&d.d_y2,         sizeof(float) * B * S),        "alloc d_y2");

  check_cuda(cudaMalloc(&d.d_logits,     sizeof(float) * B * NUM_CLASSES), "alloc d_logits");
  check_cuda(cudaMalloc(&d.d_probs,      sizeof(float) * B * NUM_CLASSES), "alloc d_probs");
  check_cuda(cudaMalloc(&d.d_grad_logits,sizeof(float) * B * NUM_CLASSES), "alloc d_grad_logits");
  check_cuda(cudaMalloc(&d.d_grad_y,     sizeof(float) * B * S),           "alloc d_grad_y");

  check_cuda(cudaMalloc(&d.d_grad_freq,  sizeof(cufftComplex) * B * S), "alloc d_grad_freq");
  check_cuda(cudaMalloc(&d.d_grad_field, sizeof(cufftComplex) * B * S), "alloc d_grad_field");
  check_cuda(cudaMalloc(&d.d_grad_freq2, sizeof(cufftComplex) * B * S), "alloc d_grad_freq2");
  check_cuda(cudaMalloc(&d.d_grad_field2,sizeof(cufftComplex) * B * S), "alloc d_grad_field2");

  check_cuda(cudaMalloc(&d.d_A,          sizeof(float) * S), "alloc d_A");
  check_cuda(cudaMalloc(&d.d_P,          sizeof(float) * S), "alloc d_P");
  check_cuda(cudaMalloc(&d.d_cosP,       sizeof(float) * S), "alloc d_cosP");
  check_cuda(cudaMalloc(&d.d_sinP,       sizeof(float) * S), "alloc d_sinP");
  check_cuda(cudaMalloc(&d.d_A2,         sizeof(float) * S), "alloc d_A2");
  check_cuda(cudaMalloc(&d.d_P2,         sizeof(float) * S), "alloc d_P2");
  check_cuda(cudaMalloc(&d.d_cosP2,      sizeof(float) * S), "alloc d_cosP2");
  check_cuda(cudaMalloc(&d.d_sinP2,      sizeof(float) * S), "alloc d_sinP2");

  check_cuda(cudaMalloc(&d.d_g_a_raw,    sizeof(float) * S), "alloc d_g_a_raw");
  check_cuda(cudaMalloc(&d.d_g_p_raw,    sizeof(float) * S), "alloc d_g_p_raw");
  check_cuda(cudaMalloc(&d.d_g_a2_raw,   sizeof(float) * S), "alloc d_g_a2_raw");
  check_cuda(cudaMalloc(&d.d_g_p2_raw,   sizeof(float) * S), "alloc d_g_p2_raw");
  check_cuda(cudaMalloc(&d.d_g_W,        sizeof(float) * NUM_CLASSES * S), "alloc d_g_W");
  check_cuda(cudaMalloc(&d.d_g_b,        sizeof(float) * NUM_CLASSES),     "alloc d_g_b");
}

void free_device_buffers(DeviceBuffers& d) {
  cudaFree(d.d_batch_in);
  cudaFree(d.d_batch_lbl);
  cudaFree(d.d_field);
  cudaFree(d.d_freq);
  cudaFree(d.d_field2);
  cudaFree(d.d_freq2);
  cudaFree(d.d_I);
  cudaFree(d.d_y);
  cudaFree(d.d_I2);
  cudaFree(d.d_y2);
  cudaFree(d.d_logits);
  cudaFree(d.d_probs);
  cudaFree(d.d_grad_logits);
  cudaFree(d.d_grad_y);
  cudaFree(d.d_grad_freq);
  cudaFree(d.d_grad_field);
  cudaFree(d.d_grad_freq2);
  cudaFree(d.d_grad_field2);
  cudaFree(d.d_A);
  cudaFree(d.d_P);
  cudaFree(d.d_cosP);
  cudaFree(d.d_sinP);
  cudaFree(d.d_A2);
  cudaFree(d.d_P2);
  cudaFree(d.d_cosP2);
  cudaFree(d.d_sinP2);
  cudaFree(d.d_g_a_raw);
  cudaFree(d.d_g_p_raw);
  cudaFree(d.d_g_a2_raw);
  cudaFree(d.d_g_p2_raw);
  cudaFree(d.d_g_W);
  cudaFree(d.d_g_b);
  std::memset(&d, 0, sizeof(d));
}

void init_params(OpticalParams& p, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> N01(0.f, 0.02f);

  p.a_raw.resize(IMG_SIZE);
  p.p_raw.resize(IMG_SIZE);
  p.a2_raw.resize(IMG_SIZE);
  p.p2_raw.resize(IMG_SIZE);
  for (int i = 0; i < IMG_SIZE; ++i) {
    p.a_raw[i] = N01(rng);
    p.p_raw[i] = N01(rng);
    p.a2_raw[i] = N01(rng);
    p.p2_raw[i] = N01(rng);
  }

  p.W.resize(NUM_CLASSES * IMG_SIZE);
  p.b.resize(NUM_CLASSES, 0.f);
  for (auto& w : p.W) w = N01(rng);

  // Adam moments
  p.ma_a.assign(IMG_SIZE, 0.f); p.va_a.assign(IMG_SIZE, 0.f);
  p.ma_p.assign(IMG_SIZE, 0.f); p.va_p.assign(IMG_SIZE, 0.f);
  p.ma_W.assign(NUM_CLASSES*IMG_SIZE, 0.f); p.va_W.assign(NUM_CLASSES*IMG_SIZE, 0.f);
  p.ma_b.assign(NUM_CLASSES, 0.f); p.va_b.assign(NUM_CLASSES, 0.f);
  p.ma_a2.assign(IMG_SIZE, 0.f); p.va_a2.assign(IMG_SIZE, 0.f);
  p.ma_p2.assign(IMG_SIZE, 0.f); p.va_p2.assign(IMG_SIZE, 0.f);

  // Initialize EMA with zeros; lazily filled on first update
  p.ema_a.clear(); p.ema_p.clear(); p.ema_a2.clear(); p.ema_p2.clear(); p.ema_W.clear(); p.ema_b.clear();
}

void create_fft_plan(FFTPlan& fft, int B) {
  fft.batch = B;
  // 2D FFT plan for IMG_H x IMG_W, batch B
  int n[2] = { IMG_H, IMG_W };
  int inembed[2]  = { IMG_H, IMG_W };
  int onembed[2]  = { IMG_H, IMG_W };
  int istride = 1, ostride = 1;
  int idist = IMG_SIZE, odist = IMG_SIZE;

  if (cufftPlanMany(&fft.plan_fwd, 2, n, inembed, istride, idist,
                                   onembed, ostride, odist,
                                   CUFFT_C2C, B) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] Failed to create forward plan\n"; std::exit(1);
  }
  if (cufftPlanMany(&fft.plan_inv, 2, n, inembed, istride, idist,
                                   onembed, ostride, odist,
                                   CUFFT_C2C, B) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] Failed to create inverse plan\n"; std::exit(1);
  }
}

void destroy_fft_plan(FFTPlan& fft) {
  cufftDestroy(fft.plan_fwd);
  cufftDestroy(fft.plan_inv);
  fft.batch = 0;
}

// ------------------------- Adam update (host) -------------------------
static inline void adam_update(std::vector<float>& param,
                               std::vector<float>& m,
                               std::vector<float>& v,
                               const float* g_dev, int n,
                               float lr, int t,
                               float beta1=0.9f, float beta2=0.999f, float eps=1e-8f,
                               float weight_decay=0.f, float grad_clip=0.f) {
  // Copy grads to host once per step (small: 784 or 7840)
  std::vector<float> g(n);
  cudaMemcpy(g.data(), g_dev, sizeof(float)*n, cudaMemcpyDeviceToHost);

  float b1t = 1.f - std::pow(beta1, t);
  float b2t = 1.f - std::pow(beta2, t);

  for (int i = 0; i < n; ++i) {
    float gi = g[i];
    if (grad_clip > 0.f) {
      if (gi >  grad_clip) gi =  grad_clip;
      if (gi < -grad_clip) gi = -grad_clip;
    }
    m[i] = beta1*m[i] + (1.f - beta1)*gi;
    v[i] = beta2*v[i] + (1.f - beta2)*gi*gi;
    float mhat = m[i] / b1t;
    float vhat = v[i] / b2t;
    float upd = mhat / (std::sqrt(vhat) + eps);
    upd += weight_decay * param[i];
    param[i] -= lr * upd;
  }
}

// ------------------------- Training batch -------------------------
float train_batch(const float* h_batch_in, const int* h_batch_lbl,
                  int B,
                  OpticalParams& params,
                  DeviceBuffers& d,
                  FFTPlan& fft,
                  float lr,
                  int t_adam,
                  bool /*use_stage2*/) {
  using utils::check_cuda;
  const int S = IMG_SIZE;
  const int C = NUM_CLASSES;

  // Upload input & labels
  check_cuda(cudaMemcpy(d.d_batch_in, h_batch_in, sizeof(float)*B*S, cudaMemcpyHostToDevice), "H2D batch_in");
  check_cuda(cudaMemcpy(d.d_batch_lbl, h_batch_lbl, sizeof(int)*B,   cudaMemcpyHostToDevice), "H2D labels");

  // Clear grads
  check_cuda(cudaMemset(d.d_g_a_raw,  0, sizeof(float)*S),   "memset g_a");
  check_cuda(cudaMemset(d.d_g_p_raw,  0, sizeof(float)*S),   "memset g_p");
  check_cuda(cudaMemset(d.d_g_a2_raw, 0, sizeof(float)*S),   "memset g_a2");
  check_cuda(cudaMemset(d.d_g_p2_raw, 0, sizeof(float)*S),   "memset g_p2");
  check_cuda(cudaMemset(d.d_g_W,     0, sizeof(float)*C*S), "memset g_W");
  check_cuda(cudaMemset(d.d_g_b,     0, sizeof(float)*C),   "memset g_b");

  // Copy params (a_raw, p_raw, a2_raw, p2_raw, W, b) to device temporaries as needed
  float *d_a_raw=nullptr, *d_p_raw=nullptr, *d_a2_raw=nullptr, *d_p2_raw=nullptr, *d_W=nullptr, *d_b=nullptr;
  check_cuda(cudaMalloc(&d_a_raw, sizeof(float)*S), "alloc d_a_raw tmp");
  check_cuda(cudaMalloc(&d_p_raw, sizeof(float)*S), "alloc d_p_raw tmp");
  check_cuda(cudaMalloc(&d_a2_raw, sizeof(float)*S), "alloc d_a2_raw tmp");
  check_cuda(cudaMalloc(&d_p2_raw, sizeof(float)*S), "alloc d_p2_raw tmp");
  check_cuda(cudaMalloc(&d_W,     sizeof(float)*C*S), "alloc d_W tmp");
  check_cuda(cudaMalloc(&d_b,     sizeof(float)*C),   "alloc d_b tmp");

  check_cuda(cudaMemcpy(d_a_raw, params.a_raw.data(), sizeof(float)*S, cudaMemcpyHostToDevice), "H2D a_raw");
  check_cuda(cudaMemcpy(d_p_raw, params.p_raw.data(), sizeof(float)*S, cudaMemcpyHostToDevice), "H2D p_raw");
  check_cuda(cudaMemcpy(d_a2_raw, params.a2_raw.data(), sizeof(float)*S, cudaMemcpyHostToDevice), "H2D a2_raw");
  check_cuda(cudaMemcpy(d_p2_raw, params.p2_raw.data(), sizeof(float)*S, cudaMemcpyHostToDevice), "H2D p2_raw");
  check_cuda(cudaMemcpy(d_W,     params.W.data(),     sizeof(float)*C*S, cudaMemcpyHostToDevice), "H2D W");
  check_cuda(cudaMemcpy(d_b,     params.b.data(),     sizeof(float)*C,   cudaMemcpyHostToDevice), "H2D b");

  // 1) Build masks
  dim3 blk(256), grdA(make_grid(S, 256));
  k_build_masks<<<grdA, blk>>>(d_a_raw, d_p_raw, d.d_A, d.d_P, d.d_cosP, d.d_sinP, S);

  // 2) Modulation to complex field
  dim3 grdM(make_grid(S, 256).x, B); // grid.y = B
  k_modulate<<<grdM, blk>>>(d.d_batch_in, d.d_A, d.d_cosP, d.d_sinP, d.d_field, S);

  // 3) FFT forward
  if (cufftExecC2C(fft.plan_fwd, d.d_field, d.d_freq, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] Exec forward failed\n"; std::exit(1);
  }

  // 4) Intensity + nonlinearity -> y1
  dim3 grdI(make_grid(S, 256).x, B);
  k_intensity_nl<<<grdI, blk>>>(d.d_freq, d.d_I, d.d_y, S);

  // 4b) Second stage: build masks, modulate y1 to field2, FFT2, NL -> y2
  k_build_masks<<<grdA, blk>>>(d_a2_raw, d_p2_raw, d.d_A2, d.d_P2, d.d_cosP2, d.d_sinP2, S);
  dim3 grdM2(make_grid(S, 256).x, B);
  k_modulate<<<grdM2, blk>>>(d.d_y, d.d_A2, d.d_cosP2, d.d_sinP2, d.d_field2, S);
  if (cufftExecC2C(fft.plan_fwd, d.d_field2, d.d_freq2, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] Exec forward stage2 failed\n"; std::exit(1);
  }
  k_intensity_nl<<<grdI, blk>>>(d.d_freq2, d.d_I2, d.d_y2, S);

  // 5) Linear head (logits)
  dim3 grdL(make_grid(C, 256));
  k_linear_forward<<<grdL, blk>>>(d.d_y2, d_W, d_b, d.d_logits, B, S, C);

  // 6) Softmax + xent, collect grad_logits and mean loss
  float* d_loss = nullptr;
  check_cuda(cudaMalloc(&d_loss, sizeof(float)), "alloc loss");
  check_cuda(cudaMemset(d_loss, 0, sizeof(float)), "memset loss");
  k_softmax_xent<<<std::min(B, 1024), 1>>>(d.d_logits, d.d_batch_lbl, d.d_probs, d.d_grad_logits, B, C, d_loss);
  float h_loss = 0.f;
  check_cuda(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost), "D2H loss");
  h_loss /= static_cast<float>(B);

  // 7) Backprop to y2
  dim3 grdBy(make_grid(S, 256));
  k_linear_backward_to_y<<<grdBy, blk>>>(d.d_grad_logits, d_W, d.d_grad_y, B, S, C);

  // Accumulate grads for W,b using y2
  k_linear_accum_grads_Wb<<<grdL, blk>>>(d.d_y2, d.d_grad_logits, d.d_g_W, d.d_g_b, B, S, C);

  // 8) Backprop y2->I2
  k_backprop_y_to_I<<<make_grid(B*S, 256), blk>>>(d.d_grad_y, d.d_I2, (float*)d.d_grad_y /*grad_I2*/, B*S);
  // 9) Backprop I2->freq2
  k_backprop_I_to_freq<<<make_grid(B*S, 256), blk>>>((float*)d.d_grad_y, d.d_freq2, d.d_grad_freq2, B*S);
  // 10) IFFT2 to grad_field2 and normalize
  if (cufftExecC2C(fft.plan_inv, d.d_grad_freq2, d.d_grad_field2, CUFFT_INVERSE) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] Exec inverse stage2 failed\n"; std::exit(1);
  }
  float scale = 1.f / float(IMG_SIZE);
  k_scale_complex<<<make_grid(B*S, 256), blk>>>(d.d_grad_field2, scale, B*S);
  // Backprop to stage2 params
  k_backprop_field_to_params_exact<<<grdA, blk>>>(d.d_y, d.d_A2, d.d_cosP2, d.d_sinP2,
                                                  d_a2_raw, d_p2_raw,
                                                  d.d_grad_field2, d.d_g_a2_raw, d.d_g_p2_raw,
                                                  B, S);
  // Backprop to y1
  k_backprop_field_to_input<<<grdA, blk>>>(d.d_grad_field2, d.d_A2, d.d_cosP2, d.d_sinP2,
                                           d.d_grad_y, B, S);
  // Continue through stage1: y1->I1->freq1->field1
  k_backprop_y_to_I<<<make_grid(B*S, 256), blk>>>(d.d_grad_y, d.d_I, (float*)d.d_grad_y /*grad_I1*/, B*S);
  k_backprop_I_to_freq<<<make_grid(B*S, 256), blk>>>((float*)d.d_grad_y, d.d_freq, d.d_grad_freq, B*S);
  if (cufftExecC2C(fft.plan_inv, d.d_grad_freq, d.d_grad_field, CUFFT_INVERSE) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] Exec inverse stage1 failed\n"; std::exit(1);
  }
  k_scale_complex<<<make_grid(B*S, 256), blk>>>(d.d_grad_field, scale, B*S);
  // Params stage1
  k_backprop_field_to_params_exact<<<grdA, blk>>>(d.d_batch_in, d.d_A, d.d_cosP, d.d_sinP,
                                                  d_a_raw, d_p_raw,
                                                  d.d_grad_field, d.d_g_a_raw, d.d_g_p_raw,
                                                  B, S);

  // 12) ADAM updates on host (small params → cheap to copy grads)
  const float clip = 1.0f;
  adam_update(params.a_raw,  params.ma_a,  params.va_a,  d.d_g_a_raw,  S,   lr, t_adam, 0.9f, 0.999f, 1e-8f, 1e-4f, clip);
  adam_update(params.p_raw,  params.ma_p,  params.va_p,  d.d_g_p_raw,  S,   lr, t_adam, 0.9f, 0.999f, 1e-8f, 1e-4f, clip);
  adam_update(params.a2_raw, params.ma_a2, params.va_a2, d.d_g_a2_raw, S,   lr, t_adam, 0.9f, 0.999f, 1e-8f, 1e-4f, clip);
  adam_update(params.p2_raw, params.ma_p2, params.va_p2, d.d_g_p2_raw, S,   lr, t_adam, 0.9f, 0.999f, 1e-8f, 1e-4f, clip);
  adam_update(params.W,      params.ma_W,  params.va_W,  d.d_g_W,      C*S, lr, t_adam, 0.9f, 0.999f, 1e-8f, 5e-4f, clip);
  adam_update(params.b,      params.ma_b,  params.va_b,  d.d_g_b,      C,    lr, t_adam, 0.9f, 0.999f, 1e-8f, 0.0f,  clip);

  // EMA update
  auto ema_update = [](std::vector<float>& ema, const std::vector<float>& p, float decay) {
    if (ema.size() != p.size()) ema.assign(p.size(), 0.f);
    for (size_t i = 0; i < p.size(); ++i) ema[i] = decay * ema[i] + (1.f - decay) * p[i];
  };
  const float ema_decay = 0.999f;
  ema_update(params.ema_a,  params.a_raw,  ema_decay);
  ema_update(params.ema_p,  params.p_raw,  ema_decay);
  ema_update(params.ema_a2, params.a2_raw, ema_decay);
  ema_update(params.ema_p2, params.p2_raw, ema_decay);
  ema_update(params.ema_W,  params.W,      ema_decay);
  ema_update(params.ema_b,  params.b,      ema_decay);

  // Free temporaries
  cudaFree(d_a_raw); cudaFree(d_p_raw); cudaFree(d_a2_raw); cudaFree(d_p2_raw); cudaFree(d_W); cudaFree(d_b); cudaFree(d_loss);

  return h_loss;
}

// ------------------------- Inference -------------------------
void infer_batch(const float* h_batch_in, int B,
                 const OpticalParams& params,
                 DeviceBuffers& d,
                 FFTPlan& fft,
                 std::vector<int>& out_labels,
                 bool /*use_stage2*/,
                 bool use_ema) {
  using utils::check_cuda;
  const int S = IMG_SIZE;
  const int C = NUM_CLASSES;

  // Upload input
  check_cuda(cudaMemcpy(d.d_batch_in, h_batch_in, sizeof(float)*B*S, cudaMemcpyHostToDevice), "H2D batch_in");

  // Upload params (read-only), optionally prefer EMA
  auto choose = [](const std::vector<float>& ema, const std::vector<float>& base) -> const std::vector<float>& {
    return (ema.size() == base.size() && !ema.empty()) ? ema : base;
  };
  const auto& a1h = use_ema ? choose(params.ema_a,  params.a_raw)  : params.a_raw;
  const auto& p1h = use_ema ? choose(params.ema_p,  params.p_raw)  : params.p_raw;
  const auto& a2h = use_ema ? choose(params.ema_a2, params.a2_raw) : params.a2_raw;
  const auto& p2h = use_ema ? choose(params.ema_p2, params.p2_raw) : params.p2_raw;
  const auto& Wh  = use_ema ? choose(params.ema_W,  params.W)      : params.W;
  const auto& bh  = use_ema ? choose(params.ema_b,  params.b)      : params.b;

  float *d_a_raw=nullptr, *d_p_raw=nullptr, *d_a2_raw=nullptr, *d_p2_raw=nullptr, *d_W=nullptr, *d_b=nullptr;
  check_cuda(cudaMalloc(&d_a_raw,  sizeof(float)*S), "alloc d_a_raw");
  check_cuda(cudaMalloc(&d_p_raw,  sizeof(float)*S), "alloc d_p_raw");
  check_cuda(cudaMalloc(&d_a2_raw, sizeof(float)*S), "alloc d_a2_raw");
  check_cuda(cudaMalloc(&d_p2_raw, sizeof(float)*S), "alloc d_p2_raw");
  check_cuda(cudaMalloc(&d_W,      sizeof(float)*C*S), "alloc d_W");
  check_cuda(cudaMalloc(&d_b,      sizeof(float)*C),   "alloc d_b");
  check_cuda(cudaMemcpy(d_a_raw,  a1h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D a_raw");
  check_cuda(cudaMemcpy(d_p_raw,  p1h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D p_raw");
  check_cuda(cudaMemcpy(d_a2_raw, a2h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D a2_raw");
  check_cuda(cudaMemcpy(d_p2_raw, p2h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D p2_raw");
  check_cuda(cudaMemcpy(d_W,      Wh.data(),  sizeof(float)*C*S, cudaMemcpyHostToDevice), "H2D W");
  check_cuda(cudaMemcpy(d_b,      bh.data(),  sizeof(float)*C,   cudaMemcpyHostToDevice), "H2D b");

  // Build masks stage 1
  dim3 blk(256), grdA((S+255)/256);
  k_build_masks<<<grdA, blk>>>(d_a_raw, d_p_raw, d.d_A, d.d_P, d.d_cosP, d.d_sinP, S);

  // Modulate stage 1
  dim3 grdM((S+255)/256, B);
  k_modulate<<<grdM, blk>>>(d.d_batch_in, d.d_A, d.d_cosP, d.d_sinP, d.d_field, S);

  // FFT stage 1
  if (cufftExecC2C(fft.plan_fwd, d.d_field, d.d_freq, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] forward failed (infer)\n"; std::exit(1);
  }

  // Intensity + NL -> y1
  dim3 grdI((S+255)/256, B);
  k_intensity_nl<<<grdI, blk>>>(d.d_freq, d.d_I, d.d_y, S);

  // Stage 2
  k_build_masks<<<grdA, blk>>>(d_a2_raw, d_p2_raw, d.d_A2, d.d_P2, d.d_cosP2, d.d_sinP2, S);
  dim3 grdM2((S+255)/256, B);
  k_modulate<<<grdM2, blk>>>(d.d_y, d.d_A2, d.d_cosP2, d.d_sinP2, d.d_field2, S);
  if (cufftExecC2C(fft.plan_fwd, d.d_field2, d.d_freq2, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] forward2 failed (infer)\n"; std::exit(1);
  }
  k_intensity_nl<<<grdI, blk>>>(d.d_freq2, d.d_I2, d.d_y2, S);

  // Linear head on y2
  dim3 grdL((C+255)/256);
  k_linear_forward<<<grdL, blk>>>(d.d_y2, d_W, d_b, d.d_logits, B, S, C);

  // Download logits and compute argmax on host (small)
  std::vector<float> h_logits(B*C);
  check_cuda(cudaMemcpy(h_logits.data(), d.d_logits, sizeof(float)*B*C, cudaMemcpyDeviceToHost), "D2H logits");
  out_labels.resize(B);
  for (int b = 0; b < B; ++b) {
    int best = 0; float bestv = h_logits[b*C + 0];
    for (int c = 1; c < C; ++c) {
      float v = h_logits[b*C + c];
      if (v > bestv) { bestv = v; best = c; }
    }
    out_labels[b] = best;
  }

  cudaFree(d_a_raw); cudaFree(d_p_raw); cudaFree(d_a2_raw); cudaFree(d_p2_raw); cudaFree(d_W); cudaFree(d_b);
}

// Return logits for TTA averaging
void infer_batch_logits(const float* h_batch_in, int B,
                        const OpticalParams& params,
                        DeviceBuffers& d,
                        FFTPlan& fft,
                        std::vector<float>& out_logits,
                        bool use_stage2,
                        bool use_ema) {
  using utils::check_cuda;
  const int S = IMG_SIZE;
  const int C = NUM_CLASSES;

  // Upload input
  check_cuda(cudaMemcpy(d.d_batch_in, h_batch_in, sizeof(float)*B*S, cudaMemcpyHostToDevice), "H2D batch_in");

  // Upload params
  auto choose = [](const std::vector<float>& ema, const std::vector<float>& base) -> const std::vector<float>& {
    return (ema.size() == base.size() && !ema.empty()) ? ema : base;
  };
  const auto& a1h = use_ema ? choose(params.ema_a,  params.a_raw)  : params.a_raw;
  const auto& p1h = use_ema ? choose(params.ema_p,  params.p_raw)  : params.p_raw;
  const auto& a2h = use_ema ? choose(params.ema_a2, params.a2_raw) : params.a2_raw;
  const auto& p2h = use_ema ? choose(params.ema_p2, params.p2_raw) : params.p2_raw;
  const auto& Wh  = use_ema ? choose(params.ema_W,  params.W)      : params.W;
  const auto& bh  = use_ema ? choose(params.ema_b,  params.b)      : params.b;
  float *d_a_raw=nullptr, *d_p_raw=nullptr, *d_a2_raw=nullptr, *d_p2_raw=nullptr, *d_W=nullptr, *d_b=nullptr;
  check_cuda(cudaMalloc(&d_a_raw,  sizeof(float)*S), "alloc d_a_raw");
  check_cuda(cudaMalloc(&d_p_raw,  sizeof(float)*S), "alloc d_p_raw");
  check_cuda(cudaMalloc(&d_a2_raw, sizeof(float)*S), "alloc d_a2_raw");
  check_cuda(cudaMalloc(&d_p2_raw, sizeof(float)*S), "alloc d_p2_raw");
  check_cuda(cudaMalloc(&d_W,      sizeof(float)*C*S), "alloc d_W");
  check_cuda(cudaMalloc(&d_b,      sizeof(float)*C),   "alloc d_b");
  check_cuda(cudaMemcpy(d_a_raw,  a1h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D a_raw");
  check_cuda(cudaMemcpy(d_p_raw,  p1h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D p_raw");
  check_cuda(cudaMemcpy(d_a2_raw, a2h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D a2_raw");
  check_cuda(cudaMemcpy(d_p2_raw, p2h.data(), sizeof(float)*S,  cudaMemcpyHostToDevice), "H2D p2_raw");
  check_cuda(cudaMemcpy(d_W,      Wh.data(),  sizeof(float)*C*S, cudaMemcpyHostToDevice), "H2D W");
  check_cuda(cudaMemcpy(d_b,      bh.data(),  sizeof(float)*C,   cudaMemcpyHostToDevice), "H2D b");

  // Stage 1
  dim3 blk(256), grdA((S+255)/256);
  k_build_masks<<<grdA, blk>>>(d_a_raw, d_p_raw, d.d_A, d.d_P, d.d_cosP, d.d_sinP, S);
  dim3 grdM((S+255)/256, B);
  k_modulate<<<grdM, blk>>>(d.d_batch_in, d.d_A, d.d_cosP, d.d_sinP, d.d_field, S);
  if (cufftExecC2C(fft.plan_fwd, d.d_field, d.d_freq, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cerr << "[cuFFT] forward failed (infer logits)\n"; std::exit(1);
  }
  dim3 grdI((S+255)/256, B);
  k_intensity_nl<<<grdI, blk>>>(d.d_freq, d.d_I, d.d_y, S);

  // Stage 2 optional
  dim3 grdL((C+255)/256);
  if (use_stage2) {
    k_build_masks<<<grdA, blk>>>(d_a2_raw, d_p2_raw, d.d_A2, d.d_P2, d.d_cosP2, d.d_sinP2, S);
    dim3 grdM2((S+255)/256, B);
    k_modulate<<<grdM2, blk>>>(d.d_y, d.d_A2, d.d_cosP2, d.d_sinP2, d.d_field2, S);
    if (cufftExecC2C(fft.plan_fwd, d.d_field2, d.d_freq2, CUFFT_FORWARD) != CUFFT_SUCCESS) {
      std::cerr << "[cuFFT] forward2 failed (infer logits)\n"; std::exit(1);
    }
    k_intensity_nl<<<grdI, blk>>>(d.d_freq2, d.d_I2, d.d_y2, S);
    k_linear_forward<<<grdL, blk>>>(d.d_y2, d_W, d_b, d.d_logits, B, S, C);
  } else {
    k_linear_forward<<<grdL, blk>>>(d.d_y, d_W, d_b, d.d_logits, B, S, C);
  }

  out_logits.resize(B*C);
  check_cuda(cudaMemcpy(out_logits.data(), d.d_logits, sizeof(float)*B*C, cudaMemcpyDeviceToHost), "D2H logits");

  cudaFree(d_a_raw); cudaFree(d_p_raw); cudaFree(d_a2_raw); cudaFree(d_p2_raw); cudaFree(d_W); cudaFree(d_b);
}
