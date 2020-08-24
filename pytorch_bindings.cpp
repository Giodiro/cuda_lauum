#include <torch/extension.h>

#include "lauum.cuh"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_lauum", &lauum, "Compute lower-LAUUM", py::arg("input"), py::arg("output"));
}
