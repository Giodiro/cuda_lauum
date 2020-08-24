import torch
import time
import numpy as np

import scipy.linalg.lapack as scll
from .cuda_lauum import cuda_lauum


def run(n):
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
    cpu_time = time.time()
    expected = scll.dlauum(matrix.numpy(), lower=1, overwrite_c=False)[0]
    cpu_time = time.time() - cpu_time

    # Run on the GPU
    gpu_time = time.time()
    cuda_lauum(gpu_in, gpu_out)
    torch.cuda.synchronize(device)
    gpu_time = time.time() - gpu_time

    # Compare outputs and print timing info
    np.testing.assert_allclose(expected, gpu_out.cpu().numpy())
    print(f"Exp. of size {n} - CPU time {cpu_time:.2f}s - GPU time {gpu_time:.2f}s")



