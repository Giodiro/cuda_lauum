import sys
import time

import torch
import numpy as np
import scipy.linalg.lapack as scll

from cuda_lauum import *



def run_gemm(n, repeat=3, dtype=torch.float32):
    torch.random.manual_seed(10)
    device = torch.device("cuda:0")
    matrix = torch.randn((n, n), dtype=dtype)
    matrix = matrix.T

    gpu_in = torch.empty_strided((n, n), stride=matrix.stride(), dtype=matrix.dtype,
                                 device=device, requires_grad=False)
    gpu_out = torch.empty_strided((n, n), stride=matrix.stride(), dtype=matrix.dtype,
                                 device=device, requires_grad=False)
    gpu_in.copy_(matrix)
    torch.cuda.synchronize()

    gpu_times = []
    for i in range(repeat):
        gpu_out.fill_(0.0)
        start_time = time.time()
        torch.mm(gpu_in, gpu_in, out=gpu_out)
        torch.cuda.synchronize()
        gpu_times.append(time.time() - start_time)
    gpu_time = min(gpu_times)
    flop = n**3 * 2
    flops = flop / gpu_time
    print(f"GEMM Exp. of size {n} - GPU time {gpu_time:.2f}s - GFlops {flops / 1e9}")


def run(n, repeat=3, compare_results=True, dtype=torch.float32, fn=cuda_lauum_lower, lower=True):
    torch.random.manual_seed(10)
    device = torch.device("cuda:0")

    # Generate random matrix
    #print("\tGenerating data...", flush=True)
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

    if compare_results:
        print("\tRunning CPU Exp...", flush=True)
        # Generate the expected output using LAPACK
        cpu_times = []
        for i in range(repeat):
            start_time = time.time()
            expected = scll.dlauum(matrix.numpy(), lower=lower, overwrite_c=False)[0]
            cpu_times.append(time.time() - start_time)
        cpu_time = min(cpu_times)
    else:
        cpu_time = 0

    #print("\tRunning GPU Exp...", flush=True)
    # Run on the GPU
    gpu_times = []
    for i in range(repeat):
        gpu_out.fill_(0.0)
        start_time = time.time()
        fn(gpu_in.shape[0], gpu_in, gpu_in.stride(1), gpu_out, gpu_out.stride(1))
        torch.cuda.synchronize(device)
        gpu_times.append(time.time() - start_time)
    gpu_time = min(gpu_times)
    flop = (2*n*(n+1)*(n+2))/6
    flops = flop / gpu_time

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
            v_cpu = np.triu(expected)
            v_gpu = np.triu(gpu_out.cpu().numpy())
            diff = np.abs(v_cpu - v_gpu)
            #with np.printoptions(precision=1, linewidth=160):
            #    print(diff)
            np.testing.assert_allclose(v_cpu, v_gpu)
    print(f"Exp. of size {n} - CPU time {cpu_time:.2f}s - GPU time {gpu_time:.2f}s  ({fn.__name__}) - GFlops {flops/1e9:.2f}")


if __name__ == "__main__":
    dt = torch.float64
    fn = cuda_lauum_lower

    #run(1000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_upper, lower=False)
    #sys.exit()

    run(1000, repeat=5, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
    time.sleep(1)
    run(5000, repeat=5, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
    time.sleep(1)
    run(10000, repeat=5, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
    time.sleep(1)
    run_gemm(10000, repeat=5, dtype=torch.float32)
    #run(10000, repeat=1, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
    #time.sleep(1)

    if False:
        for s in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]:
            run(s, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
        print()
        for s in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]:
            run(s, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower, lower=True)
            run(s, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower_square_tiled, lower=True)

    #run(1000, repeat=1, compare_results=True, dtype=torch.float64, fn=fn)
    #for s in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
    #    run(s, repeat=6, compare_results=False, dtype=dt, fn=fn)

    if False:
        #run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_upper, lower=False)
        #run(5000, repeat=1, compare_results=True, dtype=torch.float64, fn=cuda_lauum_lower)
        run_gemm(5000, repeat=3, dtype=torch.float32)
        run(5000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
        run(5000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower)
        run_gemm(10000, repeat=3, dtype=torch.float32)
        run(10000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_upper, lower=False)
        run(10000, repeat=3, compare_results=False, dtype=torch.float32, fn=cuda_lauum_lower)
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
