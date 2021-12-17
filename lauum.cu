#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>

#include "lauum.cuh"

#define BLOCK_SIZE 32
//#define DEBUG


inline int ceildiv(int dividend, int divisor) {
    int res = dividend / divisor;
    if (dividend % divisor != 0)
        res++;
    return res;
}

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
void lauum_lower_ker_sq_tiled(const scalar_t* __restrict__ in,
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


template<typename scalar_t>
__global__
void lauum_lower_ker_sq_basic(const scalar_t* __restrict__ in,
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



#define BLK_N 96
#define BLK_K 16
#define DIM_READ_X 16
#define DIM_READ_Y DIM_READ_X
#define DIM_COMP_X 16
#define DIM_COMP_Y DIM_COMP_X
#define THR_N ( BLK_N / DIM_COMP_X )
#define RC_X(val)  (val / i)
#define RC_Y(val)  (val % i)


template<typename scalar_t>
__global__
void lauum_upper_ker_tri_tiled(const scalar_t* __restrict__ in,
                               scalar_t* __restrict__ out,
                               const int size,
                               const int in_stride,
                               const int out_stride,
                               const int grid_size) {
    const int2 p = tri_index_lower(blockIdx.x);  // lower and upper are mixed up.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // DIM_COMP_X, DIM_COMP_Y  Size of the thread block for computing output
    // DIM_READ_X, DIM_READ_Y  Size of thread blocks for reading A, B
    // BLK_K, BLK_N
    // Multiplication is between two matrices of shape N, K.
    // The first dimension (N) is also referred to as X, the second (K=Y).
    __shared__ scalar_t sA[BLK_K][BLK_N];
    //__shared__ scalar_t sB[BLK_N][BLK_K + 1];
    __shared__ scalar_t sB[BLK_K][BLK_N + 1];

    //scalar_t rC[THR_N][THR_N];  // 36
    scalar_t rC[THR_N * THR_N];
    scalar_t rA[THR_N];         // 6
    scalar_t rB[THR_N];         // 6

    scalar_t ra[BLK_N / DIM_READ_X];  // 6
    scalar_t rb[BLK_N / DIM_READ_X];  // 6

    // Total work (output size) of the thread block is BLK_N * BLK_N, but
    // there are only DIM_COMP_X * DIM_COMP_Y threads. So each thread works on
    // more than a single output.
    // The thread-ids are indices of the current thread within the BLK_N, BLK_N
    // work block. Note ty goes horizontally, tx vertically.
    const int tid_global = DIM_COMP_X * ty + tx;
    
    const int tid_x = tid_global % DIM_READ_X;
    const int tid_y = tid_global / DIM_READ_X;

    int i, j, k, ki;
    int ii, jj;
    int col, row_a, row_b;

    // Zero-out rC
    # pragma unroll
    for (i = 0; i < THR_N * THR_N; i++) {
        rC[i] = 0;
    }

    // Global -> Shared (sA, sB)
    col = p.y * BLK_N + tid_y;
    # pragma unroll
    for (i = 0; i < BLK_K; i += DIM_READ_Y) {
        row_a = p.x * BLK_N + tid_x;
        row_b = p.y * BLK_N + tid_x;
        # pragma unroll
        for (j = 0; j < BLK_N; j += DIM_READ_X) {
            if (row_a <= col) {
                sA[tid_y + i][tid_x + j] = in[min(row_a + col * in_stride, size * in_stride - 1)];
            } else {
                sA[tid_y + i][tid_x + j] = 0;
            }
            if (row_b <= col) {
                sB[tid_y + i][tid_x + j] = in[min(row_b + col * in_stride, size * in_stride - 1)];
            } else {
                sB[tid_y + i][tid_x + j] = 0;
            }
            row_a += DIM_READ_X;
            row_b += DIM_READ_X;
        }
        col += DIM_READ_Y;
    }
    __syncthreads();

    for (k = p.y * BLK_N + BLK_K; k < size; k += BLK_K) {
        // Load global -> registers
        col = k + tid_y;
        row_a = p.x * BLK_N + tid_x;
        row_b = p.y * BLK_N + tid_x;
        # pragma unroll
        for (j = 0; j < BLK_N / DIM_READ_X; j++) {
            if (row_a <= col) {
                ra[j] = in[min(row_a + col * in_stride, size * in_stride - 1)];
            } else {
                ra[j] = 0;
            }
            if (row_b <= col) {
                rb[j] = in[min(row_b + col * in_stride, size * in_stride - 1)];
            } else {
                rb[j] = 0;
            }
            row_a += DIM_READ_X;
            row_b += DIM_READ_X;
        }
        // Multiply
        # pragma unroll
        for (ki = 0; ki < BLK_K; ki++) {
            // shared -> registers
            # pragma unroll
            for (i = 0; i < THR_N; i++) {
                rA[i] = sA[ki][i * DIM_COMP_X + tx];
                rB[i] = sB[ki][i * DIM_COMP_Y + ty];
            }

            // Compute
            # pragma unroll
            for (i = 0; i < THR_N * THR_N; i++) {
                rC[i] += rA[i / THR_N] * rB[i % THR_N];
            }
        }
        __syncthreads();
        // Load registers -> shared
        # pragma unroll
        for (j = 0, jj = 0; j < BLK_N; j+= DIM_READ_X, jj++) {
            sA[tid_y][tid_x + j] = ra[jj];
            sB[tid_y][tid_x + j] = rb[jj];
        }
        __syncthreads();
    }
    // Multiply last block
    # pragma unroll
    for (ki = 0; ki < BLK_K; ki++) {
        if (ki >= size - k + BLK_K)
            break;
        // shared -> registers
        # pragma unroll
        for (i = 0; i < THR_N; i++) {
            rA[i] = sA[ki][i * DIM_COMP_X + tx];
            rB[i] = sB[ki][i * DIM_COMP_Y + ty];
        }
        // Compute
        # pragma unroll
        for (i = 0; i < THR_N * THR_N; i++) {
            rC[i] += rA[i / THR_N] * rB[i % THR_N];
        }
    }

    row_a = p.x * BLK_N + tid_x;
    col = p.y * BLK_N + tid_y;
    # pragma unroll
    for (i = 0; i < THR_N * THR_N; i++) {
        // TODO: Checks for overwrite
        if ((row_a + (i / THR_N) * DIM_COMP_X) <= (col + (i % THR_N) * DIM_COMP_Y) && (col + (i % THR_N) * DIM_COMP_Y) < size) {
            out[row_a + (i / THR_N) * DIM_COMP_X + (col + (i % THR_N) * DIM_COMP_Y) * out_stride] = rC[i];
        }
    }
    /*
    # pragma unroll
    for (i = 0; i < THR_N; i++) {
        row_a = p.x * BLK_N + tid_x + i * DIM_COMP_X;
        # pragma unroll
        for (j = 0; j < THR_N; j++) {
            col = p.y * BLK_N + tid_y + j * DIM_COMP_Y;
            if (row_a <= col && col < size) {
                out[row_a + col * out_stride] = rC[i][j];
            }
        }
    }
    */

    /*
    // Initialize thread-local output (register)
    scalar_t accumulator = 0;

    for (int k = p.y; k < grid_size; k++) {
        int row_a = p.x * BLOCK_SIZE + tx;
        int col_a = k * BLOCK_SIZE + ty;
        if (row_a <= col_a && col_a < size) {
            A_tile[ty][tx] = in[row_a + col_a * in_stride];
        } else {
            A_tile[ty][tx] = 0;
        }

        int row_b = p.y * BLOCK_SIZE + tx;
        int col_b = k * BLOCK_SIZE + ty;  // Same as col_a
        if (row_b <= col_b && col_b < size) {
            B_tile[tx][ty] = in[row_b + col_b * in_stride];
        } else {
            B_tile[tx][ty] = 0;
        }
        __syncthreads();

        for (int l = 0; l < BLOCK_SIZE; l++) {
            accumulator += A_tile[l][tx] * B_tile[ty][l];
        }
        __syncthreads();
    }
    const int row_out = p.x * BLOCK_SIZE + tx;
    const int col_out = p.y * BLOCK_SIZE + ty;
    if (row_out <= col_out && col_out < size) {
        out[row_out + col_out * out_stride] = accumulator;
    } */
    /*
    // tx and ty are inverted (i.e. tx goes on along the rows,
    // ty along the columns). This allows coalesced store in the
    // write-back phase
    const int A_row = tile_pos.y * BLOCK_SIZE + tx;
    const int B_row = tile_pos.x * BLOCK_SIZE + tx;

    // Initialize shared mem
    // The first dimension of the B_tile needs to be increased to prevent bank
    // conflicts in B_tile load.


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
    }*/
}

template<typename scalar_t>
__global__
void lauum_lower_ker_tri_tiled(const scalar_t* __restrict__ in,
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


/* CPP Bindings for running the CUDA kernels */

torch::Tensor lauum_lower_square_tiled(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb) {
    const auto scalar_type = A.scalar_type();
    const int size = n;
    const int in_stride = lda;
    const int out_stride = ldb;

    const int grid_size = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_size, grid_size);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        lauum_lower_ker_sq_tiled<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_size);
    });
    return B;
}

torch::Tensor lauum_lower_square_basic(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb) {
    const auto scalar_type = A.scalar_type();
    const int size = n;
    const int in_stride = lda;
    const int out_stride = ldb;

    const int grid_size = ceildiv(size, BLOCK_SIZE);

    const dim3 dimGrid(grid_size, grid_size);
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());
        lauum_lower_ker_sq_basic<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride);
    });
    return B;
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
    //const int grid_height = ceildiv(size, BLOCK_SIZE);
    const int grid_height = ceildiv(size, BLK_N);

    const dim3 dimGrid(grid_height * (grid_height + 1) / 2, 1);
    //const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 dimBlock(DIM_COMP_X, DIM_COMP_Y);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "cuda_lauum", [&] {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        at::DeviceGuard g(A.device());

        #ifdef DEBUG
            cudaEventRecord(start);
        #endif
        lauum_upper_ker_tri_tiled<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
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
        printf("lauum_upper_ker_tri_tiled - N=%d, time=%.4fs - GFlops=%.2fGF/s\n",
               size, elapsed / 1000, (num_ops / (elapsed / 1000.0)) / 1e9);
    #endif
    return B;
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
        lauum_lower_ker_tri_tiled<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), size, in_stride, out_stride, grid_height);
    });
    return B;
}
