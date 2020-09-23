#include <torch/extension.h>

#include "lauum.cuh"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_lauum_lower", &lauum_lower, "Compute lower-LAUUM",
        py::arg("n"), py::arg("A"), py::arg("lda"), py::arg("B"), py::arg("ldb"));
  m.def("cuda_lauum_lower_square_basic", &lauum_lower_square_basic, "Compute lower-LAUUM",
        py::arg("n"), py::arg("A"), py::arg("lda"), py::arg("B"), py::arg("ldb"));
  m.def("cuda_lauum_lower_square_tiled", &lauum_lower_square_tiled, "Compute lower-LAUUM",
        py::arg("n"), py::arg("A"), py::arg("lda"), py::arg("B"), py::arg("ldb"));
}
