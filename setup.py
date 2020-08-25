from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pybind_cuda_lauum",
    version=0.1,
    description="CUDA implementation of the LAPACK LAUUM operation, and associated Python bindings",
    python_requires='~=3.6',
    ext_modules=[CUDAExtension(
        "cuda_lauum",
        sources=["pytorch_bindings.cpp", "lauum.cu"],
        include_dirs=["."],
    ), ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    include_package_data=True,
)
