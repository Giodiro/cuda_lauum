import torch
import time
import numpy as np

import scipy.linalg.lapack as scll
from cuda_lauum import cuda_lauum_lower


def run(n, repeat=3, compare_results=True):
    torch.random.manual_seed(10)
    device = torch.device("cuda:0")

    # Generate random matrix
    matrix = torch.randn((n, n), dtype=torch.float64)
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

    # Generate the expected output using LAPACK
    cpu_times = []
    for i in range(repeat):
        start_time = time.time()
        expected = scll.dlauum(matrix.numpy(), lower=1, overwrite_c=False)[0]
        cpu_times.append(time.time() - start_time)
    cpu_time = min(cpu_times)

    # Run on the GPU
    gpu_times = []
    for i in range(repeat):
        start_time = time.time()
        cuda_lauum_lower(gpu_in.shape[0], gpu_in, gpu_in.stride(1), gpu_out, gpu_out.stride(1))
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
        np.testing.assert_allclose(np.tril(expected), gpu_out.cpu().numpy())
    print(f"Exp. of size {n} - CPU time {cpu_time:.2f}s - GPU time {gpu_time:.2f}s")


if __name__ == "__main__":
    run(10000)
