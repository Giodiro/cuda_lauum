#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>


#include "lauum.cuh"
int ceildiv(int dividend, int divisor);


#define BLOCK_SIZE 32
//#define DEBUG

__device__ int2 tri_index_lower(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        linear_index - row * (row + 1) / 2,
        row
    );
}

__device__ int2 tri_index_upper(const int linear_index) {
    const int row = (int)((-1 + sqrt((double)(8*linear_index + 1))) / 2.0);
    return make_int2(
        row,
        linear_index - row * (row + 1) / 2
    );
}


template<typename scalar_t>
__global__
void lauum_square_tiled_ker2(const scalar_t* __restrict__ in,
                             scalar_t* __restrict__ out,
                             const int size,
                             const int in_stride,
                             const int out_stride,
                             const int grid_size) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int L_col = blockIdx.y * BLOCK_SIZE + ty;
    const int R_col = blockIdx.x * BLOCK_SIZE + ty;
    __shared__ scalar_t tile_L[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t tile_R[BLOCK_SIZE][BLOCK_SIZE];

    scalar_t acc = 0;
    for (int tile_i = blockIdx.y; tile_i < grid_size; tile_i++) {
	int i = tile_i * BLOCK_SIZE + tx;
	tile_L[ty][tx] = 0.0;
	tile_R[ty][tx] = 0.0;
	if (i < size && L_col <= i) {
            tile_L[ty][tx] = in[L_col * in_stride + i];
	}
	if (i < size && R_col <= i) {
            tile_R[ty][tx] = in[R_col * in_stride + i];
	}
	__syncthreads();
	for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += tile_L[k][ty] * tile_R[k][tx];
	}
	__syncthreads();
    }
    const int outx = blockIdx.x * BLOCK_SIZE + ty;
    const int outy = blockIdx.y * BLOCK_SIZE + tx;
    if (outx < size && outy < size && outx <= outy) {
        out[outx * out_stride + outy] = acc;
    }

}


template<typename scalar_t>
__global__
void lauum_square_tiled_ker(const scalar_t* __restrict__ in,
                            scalar_t* __restrict__ out,
                            const int size,
                            const int in_stride,
                            const int out_stride,
			    const int grid_size) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int L_col = blockIdx.y * BLOCK_SIZE + ty;
    const int R_col = blockIdx.x * BLOCK_SIZE + ty;
    __shared__ scalar_t tile_L[BLOCK_SIZE + 1][BLOCK_SIZE];
    __shared__ scalar_t tile_R[BLOCK_SIZE][BLOCK_SIZE];

    scalar_t acc = 0;
    for (int tile_i = blockIdx.y; tile_i < grid_size; tile_i++) {
	int i = tile_i * BLOCK_SIZE + tx;
	tile_L[ty][tx] = 0.0;
	tile_R[ty][tx] = 0.0;
	if (i < size && L_col <= i) {
            tile_L[ty][tx] = in[L_col * in_stride + i];
	}
	if (i < size && R_col <= i) {
            tile_R[ty][tx] = in[R_col * in_stride + i];
	}
	__syncthreads();
	for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += tile_L[k][ty] * tile_R[k][tx];
	}
	__syncthreads();
    }
    const int outx = blockIdx.x * BLOCK_SIZE + ty;
    const int outy = blockIdx.y * BLOCK_SIZE + tx;
    if (outx < size && outy < size && outx <= outy) {
        out[outx * out_stride + outy] = acc;
    }
}
torch::Tensor lauum_lower_square_tiled(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb) {
    const auto scalar_type = A.scalar_type();
    const int size = n;
    const int in_stride = lda;
    const int out_stride = ldb;

    // Setup CUDA grid dimensions:
    const int grid_size = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_size, grid_size);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        lauum_square_tiled_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_size);
    });
    return B;
}


template<typename scalar_t>
__global__
void lauum_square_basic_ker(const scalar_t* __restrict__ in,
                            scalar_t* __restrict__ out,
                            const int size,
                            const int in_stride,
                            const int out_stride) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int outx = blockIdx.x * blockDim.x + tx;
    const int outy = blockIdx.y * blockDim.y + ty;
    if (outy < outx || outy >= size || outx >= size) {
        return;
    }

    scalar_t acc = 0;
    for (int k = outy; k < size; k++) {
        acc += in[outy * in_stride + k] * in[outx * in_stride + k];
    }
    out[outx * out_stride + outy] = acc;
}

torch::Tensor lauum_lower_square_basic(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb) {
    const auto scalar_type = A.scalar_type();
    const int size = n;
    const int in_stride = lda;
    const int out_stride = ldb;

    // Setup CUDA grid dimensions:
    const int grid_size = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_size, grid_size);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        lauum_square_basic_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride);
    });
    return B;
}




template<typename scalar_t>
__global__
void upper_cuda_lauum_ker(const scalar_t* __restrict__ in,
                          scalar_t* __restrict__ out,
                          const int size,
                          const int in_stride,
                          const int out_stride,
                          const int grid_size) {
    const int2 tile_pos = tri_index_upper(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // tx and ty are inverted (i.e. tx goes on along the rows,
    // ty along the columns). This allows coalesced store in the
    // write-back phase
    const int A_row = tile_pos.y * BLOCK_SIZE + tx;
    const int B_row = tile_pos.x * BLOCK_SIZE + tx;

    // Initialize shared mem
    __shared__ scalar_t A_tile[BLOCK_SIZE*BLOCK_SIZE];
    // The first dimension of the B_tile needs to be increased to prevent bank
    // conflicts in B_tile load.
    __shared__ scalar_t B_tile[(BLOCK_SIZE + 1) * BLOCK_SIZE];

    // Initialize thread-local output (register)
    scalar_t accumulator = 0;

    for (int tile_i = tile_pos.x; tile_i < grid_size; tile_i++) {
        const int i = tile_i * BLOCK_SIZE + ty; 
        const int i_pos = i * in_stride;
        
        // Copy item input[row, i] and input[col, i].T to shared memory
        A_tile[ty * BLOCK_SIZE + tx] = 0;
        B_tile[tx * (BLOCK_SIZE + 1) + ty] = 0;
        if (i < size && A_row <= i) {
            A_tile[ty * BLOCK_SIZE + tx] = in[A_row + i_pos];
        }
        if (i < size && B_row <= i) {
            B_tile[tx * (BLOCK_SIZE + 1) + ty] = in[B_row + i_pos];
        }
        __syncthreads();

	#ifdef DEBUG
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - A[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, tx, ty, A_tile[tx][ty]);
        __syncthreads();
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - B[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, tx, ty, B_tile[tx][ty]);
        __syncthreads();
	#endif

        // Compute
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // Both accesses to A, B are done to prevent bank conflicts.
            // In practice we need to avoid stuff like A[tx][k] where tx is on the first dimension.
            accumulator = accumulator + A_tile[k * BLOCK_SIZE + tx] * B_tile[ty * (BLOCK_SIZE + 1) + k];
        }
        __syncthreads();
    }
    // Write-back
    const int col = tile_pos.x * BLOCK_SIZE + ty;
    const int row = tile_pos.y * BLOCK_SIZE + tx;
    if (row <= col && col < size && row < size) {
        out[row + col * out_stride] = accumulator;
    }
}

torch::Tensor lauum_upper(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb) {
    // TODO: Consistency checks

    const auto scalar_type = A.scalar_type();
    const auto size = n;
    const auto in_stride = lda;
    const auto out_stride = ldb;
#ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
#endif

    // Setup CUDA grid dimensions:
    // grid is 1D, so that we can only consider triangularly-appropriate tiles
    // blocks are 2D, with a fixed block size
    const int grid_height = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());

#ifdef DEBUG
        cudaEventRecord(start);
#endif
        upper_cuda_lauum_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_height);
#ifdef DEBUG
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#endif
    });

#ifdef DEBUG
    long long int num_ops = (2 * (long long)size * (long long)(size + 1) * (long long)(size + 2)) / 6;
    printf("num-ops: %ld  -  Flops: %.4f\n", num_ops, num_ops / (elapsed / 1000.0));
    printf("upper_cuda_lauum_ker - N=%d, time=%.4fs - GFlops=%.2fGF/s\n",
           size, elapsed / 1000, (num_ops / (elapsed / 1000.0)) / 1e9);
#endif
    return B;
}

template<typename scalar_t>
__global__
void lower_cuda_lauum_ker(const scalar_t* __restrict__ in,
                          scalar_t* __restrict__ out,
                          const int size,
                          const int in_stride,
                          const int out_stride,
                          const int grid_size) {
    // Determine the triangular tile of the output (0 indexed)
    const int2 tile_pos = tri_index_lower(blockIdx.x);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // A_col is the global column of the current thread for the A tile (transposed)
    const int A_col = tile_pos.y * BLOCK_SIZE + ty;
    // B_col is the global column of the current thread for the B tile (not transposed)
    const int B_col = tile_pos.x * BLOCK_SIZE + ty;

    // Initialize shared mem
    __shared__ scalar_t A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t B_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Initialize thread-local output (register)
    scalar_t accumulator = 0;

    for (int tile_i = tile_pos.y; tile_i < grid_size; tile_i++) {
        // i is the row position of this thread within tile-rows
        int i = tile_i * BLOCK_SIZE + tx;

        // Copy item input[i, row].T and input[i, col] to shared memory
        A_tile[ty][tx] = 0;
        B_tile[ty][tx] = 0;
        if (i < size && A_col <= i) {
            A_tile[ty][tx] = in[i + in_stride * A_col];
        }
        if (i < size && B_col <= i) {
            B_tile[ty][tx] = in[i + in_stride * B_col];
        }
        __syncthreads();

        #ifdef DEBUG
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - A[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, ty, tx, A_tile[ty][tx]);
        __syncthreads();
        printf("(tr=%d, tc=%d, ti=%d, i=%d) - B[%d, %d] = %f\n", tile_pos.y, tile_pos.x, tile_i, i, ty, tx, B_tile[ty][tx]);
        __syncthreads();
        #endif

        // Compute
        for (int k = 0; k < BLOCK_SIZE; k++) {
            accumulator = accumulator + A_tile[k][ty] * B_tile[k][tx];
        }
        __syncthreads();
    }
    // Write-back
    const int col = tile_pos.x * BLOCK_SIZE + ty;
    const int row = tile_pos.y * BLOCK_SIZE + tx;
    if (row >= col && col < size && row < size) {
        out[row + col * out_stride] = accumulator;
    }
}

torch::Tensor lauum_lower(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb) {
    // TODO: Consistency checks

    const auto scalar_type = A.scalar_type();
    const auto size = n;
    const auto in_stride = lda;
    const auto out_stride = ldb;

    // Setup CUDA grid dimensions:
    // grid is 1D, so that we can only consider triangularly-appropriate tiles
    // blocks are 2D, with a fixed block size
    const int grid_height = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        lower_cuda_lauum_ker<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_height);
    });
    return B;
}


int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}
