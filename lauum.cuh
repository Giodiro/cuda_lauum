#include <torch/extension.h>

torch::Tensor lauum_lower(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb);
torch::Tensor lauum_upper(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb);
torch::Tensor lauum_lower_square_basic(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb);
torch::Tensor lauum_lower_square_tiled(const int n, const torch::Tensor &A, const int lda, torch::Tensor &B, const int ldb);
