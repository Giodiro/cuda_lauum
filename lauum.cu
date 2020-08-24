#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>

#include "lauum.cuh"
int ceildiv(int dividend, int divisor);



template<typename scalar_t>
__global__
void lower_cuda_lauum_ker(const scalar_t *in,
                          scalar_t *out,
                          const unsigned long int size,
                          const unsigned long int grid_size) {
    // Determine the triangular tile of the output (0 indexed)
    const unsigned long int element = blockIdx.x;
    const unsigned long int tile_col = (unsigned long int)((-1 + sqrt((double)(8*element + 1))) / 2);
    const unsigned long int tile_row = element - tile_col * (tile_col + 1) / 2;

    const unsigned long int col = tile_col * blockDim.x + threadIdx.x;
    const unsigned long int row = tile_row * blockDim.y + threadIdx.y;

    // Initialize shared mem
    __shared__ scalar_t A_tile[blockDim.x][blockDim.y];
    __shared__ scalar_t B_tile[blockDim.x][blockDim.y];

    // Initialize thread-local output (register)
    scalar_t accumulator = 0;

    for (unsigned long int tile_i = tile_row; tile_i < grid_size; tile_i++) {
        // i is the row position of this thread within tile-rows
        const unsigned long int i = tile_i * blockDim.y + threadIdx.y

        // Copy item input[i, row].T and input[i, col] to shared memory
        A_tile[threadIdx.x][threadIdx.y] = in[i + row * size];
        B_tile[threadIdx.x][threadIdx.y] = in[i + col * size];
        __syncthreads();

        // Compute
        for (unsigned long k = 0; k < threadDim.x; k++) {
            accumulator = accumulator + A_tile[k][threadIdx.x] * B_tile[k][threadIdx.y];
        }
        __syncthreads();
    }

    // Write-back
    out[row + col * size] = accumulator;
}

torch::Tensor lauum(torch::Tensor &input, &output) {
    // TODO: Consistency checks

    const auto scalar_type = input.scalar_type();
    const auto size = input.size(0);

    // Setup CUDA grid dimensions:
    // grid is 1D, so that we can only consider triangularly-appropriate tiles
    // blocks are 2D, with a fixed block size
    const int block_size = 1024;
    const int grid_height = ceildiv(size, block_size);

    const dim3 dimGrid(grid_height * grid_height / 2 + grid_height);
    const dim3 dimBlock(block_size, block_size);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(input.device());
        lower_cuda_lauum_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
    });
    return output;
}


int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}
