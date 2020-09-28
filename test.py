import torch
import time
import numpy as np

import scipy.linalg.lapack as scll
from cuda_lauum import *


def run(n, repeat=3, compare_results=True, dtype=torch.float32, fn=cuda_lauum_lower, lower=True):
    torch.random.manual_seed(10)
    device = torch.device("cuda:0")

    # Generate random matrix
    print("\tGenerating data...", flush=True)
    matrix = torch.randn((n, n), dtype=dtype)
    matrix = torch.tril(matrix)
    # Make it in F-order
    matrix = matrix.T

    # Create GPU buffers for input and output matrices
    gpu_in = torch.empty_strided((n, n), stride=matrix.stride(), dtype=matrix.dtype,
                                 device=device, requires_grad=False)
    gpu_out = torch.empty_strided((n, n), stride=matrix.stride(), dtype=matrix.dtype,
                                 device=device, requires_grad=False)
    # Copy matrix to the GPU
    gpu_in.copy_(matrix)
    torch.cuda.synchronize(device)

    print("\tRunning CPU Exp...", flush=True)
    # Generate the expected output using LAPACK
    cpu_times = []
    for i in range(repeat):
        start_time = time.time()
        expected = scll.dlauum(matrix.numpy(), lower=lower, overwrite_c=False)[0]
        cpu_times.append(time.time() - start_time)
    cpu_time = min(cpu_times)

    print("\tRunning GPU Exp...", flush=True)
    # Run on the GPU
    gpu_times = []
    for i in range(repeat):
        gpu_out.fill_(0.0)
        start_time = time.time()
        fn(gpu_in.shape[0], gpu_in, gpu_in.stride(1), gpu_out, gpu_out.stride(1))
        torch.cuda.synchronize(device)
        gpu_times.append(time.time() - start_time)
    gpu_time = min(gpu_times)

    if False:
        print("INPUT")
        print(matrix)
        print("EXPECTED")
        print(torch.from_numpy(expected))
        print("ACTUAL")
        print(gpu_out)

    # Compare outputs and print timing info
    if compare_results:
        if lower:
            np.testing.assert_allclose(np.tril(expected), gpu_out.cpu().numpy())
        else:
            np.testing.assert_allclose(np.triu(expected), gpu_out.cpu().numpy())
    print(f"Exp. of size {n} - CPU time {cpu_time:.2f}s - GPU time {gpu_time:.2f}s  ({fn.__name__})")


if __name__ == "__main__":
    run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_upper, lower=False)
    run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower)
    run(5000, repeat=3, compare_results=False, dtype=torch.float64, fn=cuda_lauum_upper, lower=False)
    run(5000, repeat=3, compare_results=False, dtype=torch.float64, fn=cuda_lauum_lower)
    run(10000, repeat=3, compare_results=False, dtype=torch.float64, fn=cuda_lauum_upper, lower=False)
    run(10000, repeat=3, compare_results=False, dtype=torch.float64, fn=cuda_lauum_lower)
    if False:
        #run(4, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower_square_tiled)
        #run(50, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower_square_tiled)
        run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower)
        run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower_square_basic)
        run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower_square_tiled)
        run(5000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower)
        run(5000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower_square_basic)
        run(5000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower_square_tiled)
        run(10000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower)
        run(10000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower_square_basic)
        run(10000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower_square_tiled)
        #run(20000, repeat=3, compare_results=False, dtype=torch.float32)
        #run(30000, repeat=3, compare_results=False, dtype=torch.float32)
