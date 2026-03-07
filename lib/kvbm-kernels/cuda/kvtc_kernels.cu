// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>

namespace {

// Grid dim helper (same pattern as tensor_kernels.cu)
inline int
kvtc_compute_grid_dim(size_t total_elements, int block_dim)
{
  if (total_elements == 0)
    return 0;
  size_t blocks = (total_elements + block_dim - 1) / block_dim;
  if (blocks == 0)
    blocks = 1;
  blocks = std::min<size_t>(blocks, 65535);
  return static_cast<int>(blocks);
}

enum class TensorDataType : int { F16 = 0, BF16 = 1, F32 = 2, F64 = 3 };

template <TensorDataType>
struct DTypeTraits;
template <>
struct DTypeTraits<TensorDataType::F16> {
  using type = __half;
};
template <>
struct DTypeTraits<TensorDataType::BF16> {
  using type = __nv_bfloat16;
};
template <>
struct DTypeTraits<TensorDataType::F32> {
  using type = float;
};
template <>
struct DTypeTraits<TensorDataType::F64> {
  using type = double;
};

// ---------------------------------------------------------------------------
// Kernel 1: FP8 E4M3FN quantization (float32 -> uint8)
// ---------------------------------------------------------------------------
__global__ void
kvtc_quantize_fp8_kernel(
    const float* __restrict__ input, uint8_t* __restrict__ output, size_t batch, size_t total_features,
    size_t start_feature, size_t range_features)
{
  size_t total = batch * range_features;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * (size_t)blockDim.x) {
    size_t row = idx / range_features;
    size_t col = idx % range_features;

    float val = input[row * total_features + start_feature + col];

#if __CUDA_ARCH__ >= 890
    output[idx] = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
#else
    // Handle NaN
    if (val != val) {
      output[idx] = 0x7F;  // NaN in E4M3FN
      continue;
    }

    // Extract sign
    uint8_t sign = 0;
    if (val < 0.0f) {
      sign = 1;
      val = -val;
    }

    // Handle zero
    if (val == 0.0f) {
      output[idx] = sign << 7;
      continue;
    }

    // Clamp to max finite value (448.0)
    if (val > 448.0f) {
      val = 448.0f;
    }

    // Reinterpret as IEEE 754 float32
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(fbits));
    int fp32_exp = (int)((fbits >> 23) & 0xFF);  // biased exponent (bias=127)
    uint32_t fp32_mant = fbits & 0x7FFFFF;       // 23-bit mantissa

    // Convert exponent bias: fp32 bias=127, fp8 e4m3 bias=7
    int fp8_exp = fp32_exp - 127 + 7;

    uint8_t result;

    if (fp8_exp <= 0) {
      // Denormalized in fp8: exponent field = 0
      // Value = 2^(1-7) * (mantissa/8) = 2^(-6) * (mantissa/8)
      // We need to shift the (1.mantissa) right by (1 - fp8_exp) positions
      int shift = 1 - fp8_exp;
      // Full mantissa with implicit 1 bit, in 24-bit fixed point
      uint32_t full_mant = fp32_mant | 0x800000;  // 1.mantissa in Q23
      // We need 3 bits of mantissa for fp8; the implicit bit is at position 23
      // Shift right to align: we want the top 3 bits after shifting
      // Total right shift to get 3-bit mantissa: 23 - 3 + shift = 20 + shift
      int total_shift = 20 + shift;
      if (total_shift >= 24) {
        // Too small, rounds to zero
        output[idx] = sign << 7;
        continue;
      }
      uint32_t shifted = full_mant >> total_shift;
      // Round-to-nearest-even
      uint32_t round_bit = (total_shift > 0) ? (full_mant >> (total_shift - 1)) & 1 : 0;
      uint32_t sticky = (total_shift > 1) ? (full_mant & ((1u << (total_shift - 1)) - 1)) : 0;
      if (round_bit && (sticky || (shifted & 1))) {
        shifted += 1;
      }
      // If shifted overflows 3 bits, it becomes the smallest normal
      if (shifted >= 8) {
        result = (sign << 7) | (1 << 3) | 0;  // exp=1, mant=0
      } else {
        result = (sign << 7) | (uint8_t)shifted;
      }
    } else if (fp8_exp >= 15) {
      // Saturate to max finite: exp=14 (0b1110), mant=7 (0b111) = 0x7E unsigned
      result = (sign << 7) | 0x7E;
    } else {
      // Normalized: truncate 23-bit mantissa to 3 bits with round-to-nearest-even
      uint32_t mant3 = fp32_mant >> 20;  // top 3 bits
      uint32_t round_bit = (fp32_mant >> 19) & 1;
      uint32_t sticky = fp32_mant & 0x7FFFF;  // bottom 19 bits
      if (round_bit && (sticky || (mant3 & 1))) {
        mant3 += 1;
      }
      if (mant3 >= 8) {
        // Mantissa overflow, increment exponent
        mant3 = 0;
        fp8_exp += 1;
        if (fp8_exp >= 15) {
          // Saturate to max finite
          result = (sign << 7) | 0x7E;
        } else {
          result = (sign << 7) | ((uint8_t)fp8_exp << 3) | (uint8_t)mant3;
        }
      } else {
        result = (sign << 7) | ((uint8_t)fp8_exp << 3) | (uint8_t)mant3;
      }
    }

    output[idx] = result;
#endif
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: FP8 E4M3FN dequantization (uint8 -> float32)
// ---------------------------------------------------------------------------
__global__ void
kvtc_dequantize_fp8_kernel(
    const uint8_t* __restrict__ input, float* __restrict__ output, size_t batch, size_t total_features,
    size_t start_feature, size_t range_features)
{
  size_t total = batch * range_features;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * (size_t)blockDim.x) {
    size_t row = idx / range_features;
    size_t col = idx % range_features;

    uint8_t bits = input[idx];

#if __CUDA_ARCH__ >= 890
    // FP8 → half_raw → float (no direct fp8→float intrinsic in cuda_fp8.h)
    __half_raw hr = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
    float val = __half2float(__half(hr));
#else
    uint8_t sign_bit = (bits >> 7) & 1;
    uint8_t exp_bits = (bits >> 3) & 0xF;
    uint8_t mant_bits = bits & 0x7;

    float val;

    if (exp_bits == 0 && mant_bits == 0) {
      // Zero
      val = 0.0f;
    } else if (exp_bits == 0) {
      // Denormalized: value = 2^(-6) * (mant/8)
      val = ldexpf((float)mant_bits / 8.0f, -6);
    } else if (exp_bits == 15 && mant_bits == 7) {
      // NaN
      val = __int_as_float(0x7FC00000);  // quiet NaN
    } else {
      // Normalized: value = 2^(exp-7) * (1 + mant/8)
      val = ldexpf(1.0f + (float)mant_bits / 8.0f, (int)exp_bits - 7);
    }

    if (sign_bit) {
      val = -val;
    }
#endif

    output[row * total_features + start_feature + col] = val;
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: Min/max reduction (1 block per batch row)
// ---------------------------------------------------------------------------
__global__ void
kvtc_minmax_reduce_kernel(
    const float* __restrict__ input, float* __restrict__ min_vals, float* __restrict__ max_vals, size_t batch,
    size_t total_features, size_t start_feature, size_t range_features)
{
  // One block per row
  size_t row = blockIdx.x;
  if (row >= batch)
    return;

  const float* row_ptr = input + row * total_features + start_feature;

  extern __shared__ float sdata[];
  float* smin = sdata;               // blockDim.x floats
  float* smax = sdata + blockDim.x;  // blockDim.x floats

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  // Each thread processes multiple elements
  for (size_t i = threadIdx.x; i < range_features; i += blockDim.x) {
    float v = row_ptr[i];
    if (v < local_min)
      local_min = v;
    if (v > local_max)
      local_max = v;
  }

  smin[threadIdx.x] = local_min;
  smax[threadIdx.x] = local_max;
  __syncthreads();

  // Parallel reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (smin[threadIdx.x + s] < smin[threadIdx.x])
        smin[threadIdx.x] = smin[threadIdx.x + s];
      if (smax[threadIdx.x + s] > smax[threadIdx.x])
        smax[threadIdx.x] = smax[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    min_vals[row] = smin[0];
    max_vals[row] = smax[0];
  }
}

// ---------------------------------------------------------------------------
// Kernel 4: Integer quantization with bit-packing
// ---------------------------------------------------------------------------
__global__ void
kvtc_quantize_intx_kernel(
    const float* __restrict__ input, const float* __restrict__ min_vals, const float* __restrict__ max_vals,
    uint8_t* __restrict__ output, size_t batch, size_t total_features, size_t start_feature, size_t range_features,
    int int_bits)
{
  int slots_per_byte = 8 / int_bits;
  size_t bytes_per_row = (range_features + slots_per_byte - 1) / slots_per_byte;
  size_t total_bytes = batch * bytes_per_row;
  int max_quant = (1 << int_bits) - 1;

  for (size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x; byte_idx < total_bytes;
       byte_idx += gridDim.x * (size_t)blockDim.x) {
    size_t row = byte_idx / bytes_per_row;
    size_t byte_in_row = byte_idx % bytes_per_row;

    float min_val = min_vals[row];
    float max_val = max_vals[row];
    float interval = max_val - min_val;
    if (interval < 1e-10f)
      interval = 1.0f;

    uint8_t packed_byte = 0;

    for (int slot = 0; slot < slots_per_byte; slot++) {
      size_t col = byte_in_row * slots_per_byte + slot;
      if (col >= range_features)
        break;

      float val = input[row * total_features + start_feature + col];
      float normalized = (val - min_val) / interval * (float)max_quant;
      float rounded = roundf(normalized);
      if (rounded < 0.0f)
        rounded = 0.0f;
      if (rounded > (float)max_quant)
        rounded = (float)max_quant;
      uint8_t quantized = (uint8_t)rounded;
      packed_byte |= (quantized << (slot * int_bits));
    }

    output[byte_idx] = packed_byte;
  }
}

// ---------------------------------------------------------------------------
// Kernel 5: Integer dequantization with bit-unpacking
// ---------------------------------------------------------------------------
__global__ void
kvtc_dequantize_intx_kernel(
    const uint8_t* __restrict__ input, const float* __restrict__ min_vals, const float* __restrict__ max_vals,
    float* __restrict__ output, size_t batch, size_t total_features, size_t start_feature, size_t range_features,
    int int_bits)
{
  int slots_per_byte = 8 / int_bits;
  size_t bytes_per_row = (range_features + slots_per_byte - 1) / slots_per_byte;
  size_t total_bytes = batch * bytes_per_row;
  int max_quant = (1 << int_bits) - 1;
  uint8_t mask = (uint8_t)max_quant;

  for (size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x; byte_idx < total_bytes;
       byte_idx += gridDim.x * (size_t)blockDim.x) {
    size_t row = byte_idx / bytes_per_row;
    size_t byte_in_row = byte_idx % bytes_per_row;

    float min_val = min_vals[row];
    float max_val = max_vals[row];
    float interval = max_val - min_val;
    if (interval < 1e-10f)
      interval = 1.0f;

    uint8_t packed_byte = input[byte_idx];

    for (int slot = 0; slot < slots_per_byte; slot++) {
      size_t col = byte_in_row * slots_per_byte + slot;
      if (col >= range_features)
        break;

      uint8_t quantized = (packed_byte >> (slot * int_bits)) & mask;
      float dequantized = (float)quantized * interval / (float)max_quant + min_val;
      output[row * total_features + start_feature + col] = dequantized;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 6: Gather + mean-subtract (typed blocks -> float output)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void
kvtc_gather_mean_subtract_kernel(
    const T* const* __restrict__ block_ptrs, const float* __restrict__ mean, float* __restrict__ output,
    size_t num_blocks, size_t features, size_t block_stride)
{
  size_t total = num_blocks * features;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * (size_t)blockDim.x) {
    size_t block_idx = idx / features;
    size_t col = idx % features;
    const T* block = block_ptrs[block_idx];
    float val = static_cast<float>(block[col]);
    output[idx] = val - mean[col];
  }
}

// ---------------------------------------------------------------------------
// Kernel 7: Mean-add + scatter (float input -> typed blocks)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void
kvtc_mean_add_scatter_kernel(
    const float* __restrict__ input, const float* __restrict__ mean, T* const* __restrict__ block_ptrs,
    size_t num_blocks, size_t features, size_t block_stride)
{
  size_t total = num_blocks * features;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * (size_t)blockDim.x) {
    size_t block_idx = idx / features;
    size_t col = idx % features;
    float val = input[idx] + mean[col];
    T* block = block_ptrs[block_idx];
    block[col] = static_cast<T>(val);
  }
}

// ---------------------------------------------------------------------------
// Template dispatch helpers for gather/scatter
// ---------------------------------------------------------------------------
template <typename T>
cudaError_t
kvtc_launch_gather_mean_subtract(
    const void* const* block_ptrs, const float* mean, float* output, size_t num_blocks, size_t features,
    size_t block_stride, cudaStream_t stream)
{
  size_t total = num_blocks * features;
  if (total == 0)
    return cudaSuccess;
  int block_dim = 256;
  int grid_dim = kvtc_compute_grid_dim(total, block_dim);
  kvtc_gather_mean_subtract_kernel<T><<<grid_dim, block_dim, 0, stream>>>(
      reinterpret_cast<const T* const*>(block_ptrs), mean, output, num_blocks, features, block_stride);
  return cudaGetLastError();
}

template <typename T>
cudaError_t
kvtc_launch_mean_add_scatter(
    const float* input, const float* mean, void* const* block_ptrs, size_t num_blocks, size_t features,
    size_t block_stride, cudaStream_t stream)
{
  size_t total = num_blocks * features;
  if (total == 0)
    return cudaSuccess;
  int block_dim = 256;
  int grid_dim = kvtc_compute_grid_dim(total, block_dim);
  kvtc_mean_add_scatter_kernel<T><<<grid_dim, block_dim, 0, stream>>>(
      input, mean, reinterpret_cast<T* const*>(block_ptrs), num_blocks, features, block_stride);
  return cudaGetLastError();
}

}  // anonymous namespace

// ===========================================================================
// Extern "C" launchers
// ===========================================================================

extern "C" cudaError_t
kvbm_kernels_kvtc_quantize_fp8(
    const float* input, uint8_t* output, size_t batch, size_t total_features, size_t start_feature,
    size_t range_features, cudaStream_t stream)
{
  if (!input || !output) {
    return cudaErrorInvalidValue;
  }
  size_t total = batch * range_features;
  if (total == 0) {
    return cudaSuccess;
  }
  int block_dim = 256;
  int grid_dim = kvtc_compute_grid_dim(total, block_dim);
  kvtc_quantize_fp8_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input, output, batch, total_features, start_feature, range_features);
  return cudaGetLastError();
}

extern "C" cudaError_t
kvbm_kernels_kvtc_dequantize_fp8(
    const uint8_t* input, float* output, size_t batch, size_t total_features, size_t start_feature,
    size_t range_features, cudaStream_t stream)
{
  if (!input || !output) {
    return cudaErrorInvalidValue;
  }
  size_t total = batch * range_features;
  if (total == 0) {
    return cudaSuccess;
  }
  int block_dim = 256;
  int grid_dim = kvtc_compute_grid_dim(total, block_dim);
  kvtc_dequantize_fp8_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input, output, batch, total_features, start_feature, range_features);
  return cudaGetLastError();
}

extern "C" cudaError_t
kvbm_kernels_kvtc_minmax_reduce(
    const float* input, float* min_vals, float* max_vals, size_t batch, size_t total_features, size_t start_feature,
    size_t range_features, cudaStream_t stream)
{
  if (!input || !min_vals || !max_vals) {
    return cudaErrorInvalidValue;
  }
  if (batch == 0 || range_features == 0) {
    return cudaSuccess;
  }
  int block_dim = 256;
  int grid_dim = static_cast<int>(batch);
  size_t shared_mem = 2 * block_dim * sizeof(float);
  kvtc_minmax_reduce_kernel<<<grid_dim, block_dim, shared_mem, stream>>>(
      input, min_vals, max_vals, batch, total_features, start_feature, range_features);
  return cudaGetLastError();
}

extern "C" cudaError_t
kvbm_kernels_kvtc_quantize_intx(
    const float* input, const float* min_vals, const float* max_vals, uint8_t* output, size_t batch,
    size_t total_features, size_t start_feature, size_t range_features, int int_bits, cudaStream_t stream)
{
  if (!input || !min_vals || !max_vals || !output) {
    return cudaErrorInvalidValue;
  }
  if (int_bits <= 0 || int_bits > 8 || (8 % int_bits) != 0) {
    return cudaErrorInvalidValue;
  }
  int slots_per_byte = 8 / int_bits;
  size_t bytes_per_row = (range_features + slots_per_byte - 1) / slots_per_byte;
  size_t total_bytes = batch * bytes_per_row;
  if (total_bytes == 0) {
    return cudaSuccess;
  }
  int block_dim = 256;
  int grid_dim = kvtc_compute_grid_dim(total_bytes, block_dim);
  kvtc_quantize_intx_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input, min_vals, max_vals, output, batch, total_features, start_feature, range_features, int_bits);
  return cudaGetLastError();
}

extern "C" cudaError_t
kvbm_kernels_kvtc_dequantize_intx(
    const uint8_t* input, const float* min_vals, const float* max_vals, float* output, size_t batch,
    size_t total_features, size_t start_feature, size_t range_features, int int_bits, cudaStream_t stream)
{
  if (!input || !min_vals || !max_vals || !output) {
    return cudaErrorInvalidValue;
  }
  if (int_bits <= 0 || int_bits > 8 || (8 % int_bits) != 0) {
    return cudaErrorInvalidValue;
  }
  int slots_per_byte = 8 / int_bits;
  size_t bytes_per_row = (range_features + slots_per_byte - 1) / slots_per_byte;
  size_t total_bytes = batch * bytes_per_row;
  if (total_bytes == 0) {
    return cudaSuccess;
  }
  int block_dim = 256;
  int grid_dim = kvtc_compute_grid_dim(total_bytes, block_dim);
  kvtc_dequantize_intx_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input, min_vals, max_vals, output, batch, total_features, start_feature, range_features, int_bits);
  return cudaGetLastError();
}

extern "C" cudaError_t
kvbm_kernels_kvtc_gather_mean_subtract(
    const void* const* block_ptrs, const float* mean, float* output, size_t num_blocks, size_t features,
    size_t block_stride, int input_dtype, cudaStream_t stream)
{
  if (!block_ptrs || !mean || !output)
    return cudaErrorInvalidValue;
  auto dtype = static_cast<TensorDataType>(input_dtype);
  switch (dtype) {
    case TensorDataType::F16:
      return kvtc_launch_gather_mean_subtract<typename DTypeTraits<TensorDataType::F16>::type>(
          block_ptrs, mean, output, num_blocks, features, block_stride, stream);
    case TensorDataType::BF16:
      return kvtc_launch_gather_mean_subtract<typename DTypeTraits<TensorDataType::BF16>::type>(
          block_ptrs, mean, output, num_blocks, features, block_stride, stream);
    case TensorDataType::F32:
      return kvtc_launch_gather_mean_subtract<typename DTypeTraits<TensorDataType::F32>::type>(
          block_ptrs, mean, output, num_blocks, features, block_stride, stream);
    case TensorDataType::F64:
      return kvtc_launch_gather_mean_subtract<typename DTypeTraits<TensorDataType::F64>::type>(
          block_ptrs, mean, output, num_blocks, features, block_stride, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

extern "C" cudaError_t
kvbm_kernels_kvtc_mean_add_scatter(
    const float* input, const float* mean, void* const* block_ptrs, size_t num_blocks, size_t features,
    size_t block_stride, int output_dtype, cudaStream_t stream)
{
  if (!input || !mean || !block_ptrs)
    return cudaErrorInvalidValue;
  auto dtype = static_cast<TensorDataType>(output_dtype);
  switch (dtype) {
    case TensorDataType::F16:
      return kvtc_launch_mean_add_scatter<typename DTypeTraits<TensorDataType::F16>::type>(
          input, mean, block_ptrs, num_blocks, features, block_stride, stream);
    case TensorDataType::BF16:
      return kvtc_launch_mean_add_scatter<typename DTypeTraits<TensorDataType::BF16>::type>(
          input, mean, block_ptrs, num_blocks, features, block_stride, stream);
    case TensorDataType::F32:
      return kvtc_launch_mean_add_scatter<typename DTypeTraits<TensorDataType::F32>::type>(
          input, mean, block_ptrs, num_blocks, features, block_stride, stream);
    case TensorDataType::F64:
      return kvtc_launch_mean_add_scatter<typename DTypeTraits<TensorDataType::F64>::type>(
          input, mean, block_ptrs, num_blocks, features, block_stride, stream);
    default:
      return cudaErrorInvalidValue;
  }
}
