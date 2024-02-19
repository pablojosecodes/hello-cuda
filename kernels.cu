#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b;}



__global__ void gray_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];
}

__global__ void matmul_k(float* m, float* n, float* out, int h, int w, int k) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;

    if (r>=h || c>=w) return;
    float o = 0;
    for (int i = 0; i<k; ++i) o += m[r*k+i] * n[i*w+c];
    out[r*w+c] = o;
}


__global__ void matmul_kernel(float* m, float* n, float* out, int h, int w, int k) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;

    if (r>=h || c>=w) return;
    float o = 0;
    for (int i = 0; i<k; ++i) o += m[r*k+i] * n[i*w+c];
    out[r*w+c] = o;
}

__global__ void matmul_tiled_kernel(float* m, float* n, float* out, int h, int w, int k, int tw) {
    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int r = blockIdx.y * blockDim.y + tr;
    int c = blockIdx.x * blockDim.x + tc;

    // Load shared (within block) memoryh
    extern __shared__ float shared[];

    float* ms = &shared[0];
    float* ns = &shared[tw * tw];

    float p = 0.0;
    for (int ph = 0; ph < cdiv(k, tw); ++ph) {

        // Calculate the shared memory
        ms[tr * tw + tc] = (r < h && (ph * tw + tc) < k) ? m[tc + ph * tw + r * k] : 0.0;
        ns[tr * tw + tc] = (c < w && (ph * tw + tr) < k) ? n[(tr + ph * tw) * w + c] : 0.0;

        // Sync up with other threads in block
        __syncthreads();

        // Utilize shared memory
        for (int i = 0; i < tw; ++i) p += ms[tr * tw + i] * ns[tw * i + tc];
        __syncthreads();
    }

    if (r < h && c < w) out[r * w + c] = p;
}



