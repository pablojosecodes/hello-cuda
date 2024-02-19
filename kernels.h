#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Utility macros and functions
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b);

// Declare the host-side wrapper functions for CUDA kernels
torch::Tensor grayscale_cuda(torch::Tensor x);
torch::Tensor matmul_cuda(torch::Tensor m, torch::Tensor n);
torch::Tensor matmul_tiled_cuda(torch::Tensor m, torch::Tensor n, int tw);
