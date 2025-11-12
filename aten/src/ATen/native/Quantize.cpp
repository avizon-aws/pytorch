#include <ATen/ATen.h>
#include <ATen/native/Quantize.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <tuple>

namespace at {
namespace native {

namespace {

// Helper function for rounding
template <typename scalar_t>
inline scalar_t apply_rounding(scalar_t value, int64_t rounding_mode, uint32_t* rand_state = nullptr) {
  switch (rounding_mode) {
    case 0: // floor
      return std::floor(value);
    case 1: // round to nearest even
      return std::rint(value);
    case 2: // stochastic (simplified)
      if (rand_state) {
        float frac = value - std::floor(value);
        // Simple LCG random number generator
        *rand_state = (*rand_state * 1103515245 + 12345) & 0x7fffffff;
        float rand_val = static_cast<float>(*rand_state) / 0x7fffffff;
        return (rand_val < frac) ? std::ceil(value) : std::floor(value);
      }
      return std::rint(value);
    default:
      return std::rint(value);
  }
}

// FP8 quantization parameters
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_E5M2_MAX = 57344.0f;

float get_fp8_max(at::ScalarType dtype) {
  if (dtype == at::kFloat8_e4m3fn) {
    return FP8_E4M3_MAX;
  } else if (dtype == at::kFloat8_e5m2) {
    return FP8_E5M2_MAX;
  }
  return FP8_E4M3_MAX;
}

template <typename input_t, typename output_t>
void quantize_fp8_impl(
    const input_t* input_data,
    output_t* output_data,
    float* scale_data,
    int64_t numel,
    int64_t block_size,
    float fp8_max,
    int64_t rounding_mode) {
  
  int64_t num_blocks = (numel + block_size - 1) / block_size;
  
  at::parallel_for(0, num_blocks, 0, [&](int64_t begin, int64_t end) {
    uint32_t rand_state = begin + 12345; // Simple seed
    
    for (int64_t block_idx = begin; block_idx < end; ++block_idx) {
      int64_t start = block_idx * block_size;
      int64_t block_end = std::min(start + block_size, numel);
      
      // Find max absolute value in block
      float amax = 0.0f;
      for (int64_t i = start; i < block_end; ++i) {
        float abs_val = std::abs(static_cast<float>(input_data[i]));
        amax = std::max(amax, abs_val);
      }
      
      // Compute scale
      float scale = (amax == 0.0f) ? 1.0f : (fp8_max / amax);
      scale_data[block_idx] = 1.0f / scale; // Store inverse scale for dequant
      
      // Quantize block
      for (int64_t i = start; i < block_end; ++i) {
        float val = static_cast<float>(input_data[i]) * scale;
        val = std::max(-fp8_max, std::min(val, fp8_max));
        output_data[i] = static_cast<output_t>(val);
      }
    }
  });
}

template <typename input_t, typename output_t>
void quantize_mxfp8_impl(
    const input_t* input_data,
    output_t* output_data,
    float* scale_data,
    int64_t numel,
    int64_t block_size,
    float fp8_max,
    int64_t rounding_mode) {
  
  int64_t num_blocks = (numel + block_size - 1) / block_size;
  
  at::parallel_for(0, num_blocks, 0, [&](int64_t begin, int64_t end) {
    for (int64_t block_idx = begin; block_idx < end; ++block_idx) {
      int64_t start = block_idx * block_size;
      int64_t block_end = std::min(start + block_size, numel);
      
      // Find max absolute value in block
      float amax = 0.0f;
      for (int64_t i = start; i < block_end; ++i) {
        float abs_val = std::abs(static_cast<float>(input_data[i]));
        amax = std::max(amax, abs_val);
      }
      
      // Compute shared exponent for MXFP
      int shared_exp = 0;
      if (amax > 0.0f) {
        shared_exp = static_cast<int>(std::floor(std::log2(amax))) + 1;
      }
      
      float scale = std::pow(2.0f, -shared_exp);
      scale_data[block_idx] = std::pow(2.0f, shared_exp); // Store for dequant
      
      // Quantize block with shared exponent
      for (int64_t i = start; i < block_end; ++i) {
        float val = static_cast<float>(input_data[i]) * scale;
        val = std::max(-fp8_max, std::min(val, fp8_max));
        output_data[i] = static_cast<output_t>(val);
      }
    }
  });
}

} // anonymous namespace

std::tuple<Tensor, Tensor> quantize_cpu(
    const Tensor& self,
    int64_t block_size,
    at::ScalarType dtype,
    int64_t quant_type,
    int64_t rounding_mode) {
  
  TORCH_CHECK(self.dim() >= 1, "Input tensor must have at least 1 dimension");
  TORCH_CHECK(block_size > 0, "Block size must be positive");
  TORCH_CHECK(quant_type >= 0 && quant_type <= 1, "quant_type must be 0 (FP8) or 1 (MXFP8)");
  TORCH_CHECK(rounding_mode >= 0 && rounding_mode <= 2, "rounding_mode must be 0, 1, or 2");
  
  auto output = at::empty_like(self, self.options().dtype(dtype));
  
  int64_t numel = self.numel();
  int64_t num_blocks = (numel + block_size - 1) / block_size;
  auto scale = at::empty({num_blocks}, self.options().dtype(at::kFloat));
  
  float fp8_max = get_fp8_max(dtype);
  
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, self.scalar_type(), "quantize_cpu", [&] {
    const scalar_t* input_data = self.data_ptr<scalar_t>();
    float* scale_data = scale.data_ptr<float>();
    
    if (dtype == at::kFloat8_e4m3fn) {
      auto* output_data = output.data_ptr<at::Float8_e4m3fn>();
      if (quant_type == 0) {
        quantize_fp8_impl(input_data, output_data, scale_data, numel, block_size, fp8_max, rounding_mode);
      } else {
        quantize_mxfp8_impl(input_data, output_data, scale_data, numel, block_size, fp8_max, rounding_mode);
      }
    } else if (dtype == at::kFloat8_e5m2) {
      auto* output_data = output.data_ptr<at::Float8_e5m2>();
      if (quant_type == 0) {
        quantize_fp8_impl(input_data, output_data, scale_data, numel, block_size, fp8_max, rounding_mode);
      } else {
        quantize_mxfp8_impl(input_data, output_data, scale_data, numel, block_size, fp8_max, rounding_mode);
      }
    } else {
      TORCH_CHECK(false, "Unsupported dtype for quantization");
    }
  });
  
  return std::make_tuple(output, scale);
}

} // namespace native
} // namespace at