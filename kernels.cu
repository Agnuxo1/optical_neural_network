#include "optical_model.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Utility launch config ----------
static inline dim3 make_grid(int N, int block) {
  return dim3((N + block - 1) / block);
}

// ---------- Parameter transforms ----------
__global__ void k_build_masks(const float* a_raw, const float* p_raw,
                              float* A, float* P, float* cosP, float* sinP,
                              int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  // amplitude: softplus(z) = log(1 + e^z); derivative sigmoid(z)
  float ar = a_raw[i];
  float A_i = log1pf(expf(ar)) + 1e-3f; // ensure > 0
  A[i] = A_i;
  // phase: pi * tanh(z)
  float pr = p_raw[i];
  float P_i = (float)M_PI * tanhf(pr);
  P[i] = P_i;
  cosP[i] = cosf(P_i);
  sinP[i] = sinf(P_i);
}

// ---------- Build complex field: field = A * x * (cosP + i sinP) ----------
__global__ void k_modulate(const float* x, // [B, S]
                           const float* A,
                           const float* cosP,
                           const float* sinP,
                           cufftComplex* field, // [B, S]
                           int S) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S) return;
  float A_i = A[idx], c = cosP[idx], s = sinP[idx];
  // Process batch stride
  for (int b = 0; b < gridDim.y; ++b) {
    int offset = b * S + idx;
    float xv = x[offset];
    float re = A_i * xv * c;
    float im = A_i * xv * s;
    field[offset] = make_cuFloatComplex(re, im);
  }
}

// ---------- Intensity & nonlinearity ----------
__global__ void k_intensity_nl(const cufftComplex* freq, // [B, S]
                               float* I, float* y,        // [B, S]
                               int S) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= S) return;
  for (int b = 0; b < gridDim.y; ++b) {
    int o = b * S + idx;
    float ur = freq[o].x, ui = freq[o].y;
    float ii = ur*ur + ui*ui;      // |U|^2
    I[o] = ii;
    y[o] = log1pf(ii);             // compression for stability
  }
}

// ---------- Linear head (logits = W*y + b) ----------
__global__ void k_linear_forward(const float* y,     // [B, S]
                                 const float* W,     // [C, S]
                                 const float* b,     // [C]
                                 float* logits,      // [B, C]
                                 int B, int S, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  for (int bidx = 0; bidx < B; ++bidx) {
    float acc = b[c];
    const float* yptr = y + bidx * S;
    const float* Wrow = W + c * S;
    for (int i = 0; i < S; ++i) acc += Wrow[i] * yptr[i];
    logits[bidx*C + c] = acc;
  }
}

// ---------- Softmax + cross-entropy grad ----------
__global__ void k_softmax_xent(const float* logits, const int* labels,
                               float* probs, float* grad_logits,
                               int B, int C, float* loss_out) {
  // One block handles the whole batch loss accumulation (simple)
  // (Not the fastest, but sufficient for small B and C=10)
  if (threadIdx.x == 0 && blockIdx.x == 0) *loss_out = 0.f;
  __syncthreads();

  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    // Compute max-logit for stability
    float maxv = -1e30f;
    for (int c = 0; c < C; ++c) {
      float v = logits[b*C + c];
      if (v > maxv) maxv = v;
    }
    // Compute denominator
    float sum = 0.f;
    for (int c = 0; c < C; ++c) sum += expf(logits[b*C + c] - maxv);
    // Probabilities and loss
    int y = labels[b];
    float loss_b = 0.f;
    for (int c = 0; c < C; ++c) {
      float p = expf(logits[b*C + c] - maxv) / sum;
      probs[b*C + c] = p;
      float g = p - ((c == y) ? 1.f : 0.f);
      grad_logits[b*C + c] = g;
      if (c == y) loss_b = -logf(fmaxf(p, 1e-12f));
    }
    // Atomic add to total loss
    atomicAdd(loss_out, loss_b);
  }
}

// ---------- Backprop to y: grad_y = grad_logits * W ----------
__global__ void k_linear_backward_to_y(const float* grad_logits, // [B, C]
                                       const float* W,           // [C, S]
                                       float* grad_y,            // [B, S]
                                       int B, int S, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // feature idx
  if (i >= S) return;
  for (int b = 0; b < B; ++b) {
    float acc = 0.f;
    const float* gl = grad_logits + b*C;
    for (int c = 0; c < C; ++c) acc += gl[c] * W[c*S + i];
    grad_y[b*S + i] = acc;
  }
}

// ---------- Accumulate grads for W and b ----------
__global__ void k_linear_accum_grads_Wb(const float* y,            // [B, S]
                                        const float* grad_logits,  // [B, C]
                                        float* gW, float* gb,      // [C,S], [C]
                                        int B, int S, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;

  float gb_acc = 0.f;
  extern __shared__ float sdata[]; // optional scratch, not used heavily here
  (void)sdata;

  // gW[c, i] += sum_b grad_logits[b, c] * y[b, i]
  for (int i = 0; i < S; ++i) {
    float acc = 0.f;
    for (int b = 0; b < B; ++b) {
      acc += grad_logits[b*C + c] * y[b*S + i];
    }
    atomicAdd(&gW[c*S + i], acc);
  }
  for (int b = 0; b < B; ++b) gb_acc += grad_logits[b*C + c];
  atomicAdd(&gb[c], gb_acc);
}

// ---------- Backprop through y = log1p(I) ----------
__global__ void k_backprop_y_to_I(const float* grad_y, const float* I,
                                  float* grad_I, int BS) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS) return;
  grad_I[idx] = grad_y[idx] / (1.f + I[idx]);
}

// ---------- Backprop I=|U|^2 to freq (Ur, Ui) ----------
__global__ void k_backprop_I_to_freq(const float* grad_I,
                                     const cufftComplex* freq,
                                     cufftComplex* grad_freq,
                                     int BS) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS) return;
  float ur = freq[idx].x, ui = freq[idx].y;
  float gI = grad_I[idx];
  grad_freq[idx].x = 2.f * gI * ur;
  grad_freq[idx].y = 2.f * gI * ui;
}

// ---------- Scale by factor (for IFFT normalization) ----------
__global__ void k_scale_complex(cufftComplex* x, float s, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  x[i].x *= s;
  x[i].y *= s;
}

// ---------- Backprop to a_raw and p_raw ----------
__global__ void k_backprop_field_to_params(const float* x, // [B,S]
                                           const float* A,
                                           const float* cosP,
                                           const float* sinP,
                                           const cufftComplex* grad_field, // [B,S]
                                           float* g_a_raw, float* g_p_raw, // [S]
                                           int B, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S) return;

  float ga = 0.f, gp = 0.f;
  float Ai = A[i], c = cosP[i], s = sinP[i];

  for (int b = 0; b < B; ++b) {
    int o = b*S + i;
    float xv = x[o];
    float dRe = grad_field[o].x;
    float dIm = grad_field[o].y;

    // dL/dA = x*( c*dRe + s*dIm )
    float dLdA = xv * (c * dRe + s * dIm);
    // dL/dP = A*x*( -s*dRe + c*dIm )
    float dLdP = Ai * xv * (-s * dRe + c * dIm);

    // chain: A = softplus(a_raw) + eps -> dA/da_raw = sigmoid(a_raw)
    // We don't have a_raw here; approximate sigmoid from A:
    // softplus(ar) = log(1+e^ar) -> not invertible cheaply; instead:
    // Use stable surrogate: dA_da ≈ clamp(A / (1 + A), 0, 1) (works well in practice for >0)
    float dA_da = fminf(fmaxf(Ai / (1.f + Ai), 0.f), 1.f);
    // P = pi * tanh(p_raw) -> dP/dp_raw = pi*(1 - tanh^2) ; we don't know tanh(p_raw), but
    // |P| <= pi, approximate by: dP_dp ≈ (pi - |P|) / pi  (smooth surrogate)
    float Pi = atan2f(s, c); // recover P from cos/sin (principal value)
    float dP_dp = (float)M_PI > 0.f ? fmaxf(0.f, ((float)M_PI - fabsf(Pi)) / (float)M_PI) : 0.f;

    ga += dLdA * dA_da;
    gp += dLdP * dP_dp;
  }
  atomicAdd(&g_a_raw[i], ga);
  atomicAdd(&g_p_raw[i], gp);
}

// ---------- Backprop to a_raw and p_raw (exact chain) ----------
__global__ void k_backprop_field_to_params_exact(const float* x, // [B,S]
                                                 const float* A,
                                                 const float* cosP,
                                                 const float* sinP,
                                                 const float* a_raw,   // [S]
                                                 const float* p_raw,   // [S]
                                                 const cufftComplex* grad_field, // [B,S]
                                                 float* g_a_raw, float* g_p_raw, // [S]
                                                 int B, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S) return;

  float ga = 0.f, gp = 0.f;
  float Ai = A[i], c = cosP[i], s = sinP[i];

  for (int b = 0; b < B; ++b) {
    int o = b*S + i;
    float xv = x[o];
    float dRe = grad_field[o].x;
    float dIm = grad_field[o].y;

    // dL/dA = x*( c*dRe + s*dIm )
    float dLdA = xv * (c * dRe + s * dIm);
    // dL/dP = A*x*( -s*dRe + c*dIm )
    float dLdP = Ai * xv * (-s * dRe + c * dIm);

    // Exact chains using raw params
    float ar = a_raw[i];
    float dA_da = 1.f / (1.f + expf(-ar));
    float pr = p_raw[i];
    float th = tanhf(pr);
    float dP_dp = (float)M_PI * (1.f - th * th);

    ga += dLdA * dA_da;
    gp += dLdP * dP_dp;
  }
  atomicAdd(&g_a_raw[i], ga);
  atomicAdd(&g_p_raw[i], gp);
}

// ---------- Backprop to input x through field = A*x*(c + i s) ----------
__global__ void k_backprop_field_to_input(const cufftComplex* grad_field, // [B,S]
                                          const float* A,
                                          const float* cosP,
                                          const float* sinP,
                                          float* grad_x, // [B,S]
                                          int B, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S) return;
  float Ai = A[i], c = cosP[i], s = sinP[i];
  for (int b = 0; b < B; ++b) {
    int o = b*S + i;
    float dRe = grad_field[o].x;
    float dIm = grad_field[o].y;
    grad_x[o] = Ai * (c * dRe + s * dIm);
  }
}
