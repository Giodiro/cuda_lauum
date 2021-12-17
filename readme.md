# CUDA LAUUM

CUDA implementation of the LAUUM LAPACK function (triangular matrix multiplication).

## LAUUM

The LAUUM function computes the product ![U U^\top](https://latex.codecogs.com/png.latex?\bg_white&space;\inline&space;U&space;U^\top) or 
![L^\top L](https://latex.codecogs.com/png.latex?\bg_white&space;\inline&space;L^\top&space;L) where ![U](https://latex.codecogs.com/png.latex?\bg_white&space;\inline&space;U) is an **upper triangular** matrix and 
![L](https://latex.codecogs.com/png.latex?\bg_white&space;\inline&space;L) is a **lower triangular** matrix. Since the result is symmetric, the output will also be a triangular matrix.

The complexity of the algorithm is cubic in the matrix size (as with all matrix multiplication), but some performance can be gained
by ignoring the lower/upper triangular parts of the input. The overall complexity for a matrix of size n is of 
![\frac{n \times (n + 1) \times (n + 2)}{3}](https://latex.codecogs.com/png.latex?\bg_white&space;\frac{n&space;\times&space;(n&space;&plus;&space;1)&space;\times&space;(n&space;&plus;&space;2)}{3})

## Implementations

The repository contains a few implementations for pedagogical purposes, ranging from the the most naive, 
to a relatively complex implementation which reaches a decent performance. The input matrices must be stored in column contiguous order.

The implementations are linked to Python via PyTorch extensions for ease of use. 
PRs are welcome to make the code more usable from other programming languages.


## Benchmarks

The final implementation was tested on two different GPUs: a NVIDIA Titan Xp (compute capability 6.1) 
and a Quadro RTX 6000 (compute capability 7.5).

We compare the running time and calculated GFlops for our LAUUM implementation, and the matrix multiplication routine (GEMM) of cuBLAS.
The benchmarks show our implementation reaches between one half and one third of the maximum attainable performance (assuming GEMM achieves close to peak performance of the card).

All experiments were run with single precision floats.

| n | LAUUM (Titan Xp) | cuBLAS GEMM (Titan Xp) | LAUUM (RTX 6000) | cuBLAS GEMM (RTX 6000) |
| - | ---------------- | ---------------------- | ---------------- | ---------------------- |
| 1000  | 1632 GFlops  | 7997 GFlops  | 1279 GFlops | 9754 GFlops  |
| 5000  | 5798 GFlops  | 9221 GFlops  | 5537 GFlops | 12332 GFlops |
| 10000 | 6713 GFlops  | 10657 GFlops | 5162 GFlops | 13709 GFlops |

Clearly there is still some margin for improvement on our implementation, especially on the newer cards.


## Installation & Usage

This is not meant to be production-ready code, hence to install you can go the quick and dirty way

```bash
git clone https://github.com/Giodiro/cuda_lauum.git
cd cuda_lauum
python setup.py develop  # or pip install -e .
```

A simple usage example:
```python
from cuda_lauum import cuda_lauum_lower, cuda_lauum_upper

# Generate random matrix
matrix = torch.randn((n, n), dtype=torch.float32, device="cuda:0")
# Make it in F-order
matrix = matrix.T
# Create the output matrix
output = torch.zeros_like(matrix)
# Run LAUUM. The output will be stored in the upper triangle of `output`.
cuda_lauum_upper(matrix.size(0), matrix, matrix.stride(1), output, output.stride(1))
```


## TODO List

 - [ ] Improve performance on RTX 6000
 - [ ] Improve python interface
 - [ ] Remove (or move to another file) the slow implementations, since compilation is slowed down by them.
 - [ ] Implement the better kernel also for lower LAUUM. This would allow a decent interface which decides lower/upper depending on matrix order, etc.

 
