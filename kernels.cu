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


// Specifically for convolutoinal kernel (constant memory on CUDA kernel)
__constant__ float c_M[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

__device__ int two_to_one(int r, int c, int c_size) {
    return r * c_size + c;  // Adjusted for correct row-major indexing
}

__global__ void conv_kernel(const float* m, float* out, int w, int h) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < w && c < h) {
        int index = two_to_one(r, c, h);

        float c_out = 0.0f;
        int maskIndex = 0;

        for (int x = r - 1; x <= r + 1; x++) {
            for (int y = c - 1; y <= c + 1; y++) {
                if (x >= 0 && x < w && y >= 0 && y < h) {
                    c_out += m[two_to_one(x, y, h)] * c_M[maskIndex];
                }
                maskIndex++;
            }
        }

        out[index] = c_out;
    }
}

// Actually parallelizable and efficient
#define IN_TILE_DIM 32
#define FILTER_RADIUS 1
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

#define FILTER_DIM 3

__constant__ float F[FILTER_DIM][FILTER_DIM] = {
    {1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f}
};


__global__ void cuda_tiled(const float *N, float *O, int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    extern __shared__ float N_s[];

    if (0 <= row && row < height && 0 <= col && col < width) {
        N_s[threadIdx.y * IN_TILE_DIM + threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y * IN_TILE_DIM + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (0 <= col && col < width && 0 <= row && row < height) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_DIM; i++) {
            for (int j = 0; j < FILTER_DIM; j++) {
                int r = row - FILTER_RADIUS + i;
                int c = col - FILTER_RADIUS + j;
                if (0 <= r && r < height && 0 <= c && c < width) {
                    sum += N_s[(threadIdx.y - FILTER_RADIUS + i) * IN_TILE_DIM + (threadIdx.x - FILTER_RADIUS + j)] * F[i][j];
                }
            }
        }
        O[row * width + col] = sum;
    }
}


