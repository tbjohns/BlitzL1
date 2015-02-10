#include "common.h"

using std::vector;

namespace BlitzL1 {

  value_t Column::l2_norm_centered() const {
    value_t value_sq = l2_norm_sq() - mean() * mean() * length;
    if (value_sq <= 0)
      return 0;
    else
      return sqrt(value_sq);
  }

  ColumnFromPointers::ColumnFromPointers(
      index_t *indices, value_t *values, index_t nnz, index_t length) {
    this->indices = indices;
    this->values = values;
    this->nnz = nnz;
    this->length = length;
    return;
  }

  value_t ColumnFromPointers::inner_product(vector<value_t> vec) const {
    value_t result = 0.0;
    for (index_t ind = 0; ind < nnz; ++ind) {
      index_t i = indices[ind];
      result += values[ind] * vec[i];
    }
    return result;
  }

  void ColumnFromPointers::add_multiple(vector<value_t> &target, value_t scaler) const {
    for (index_t ind = 0; ind < nnz; ++ind) {
      index_t i = indices[ind];
      target[i] += values[ind] * scaler;
    }
  }

  value_t ColumnFromPointers::h_norm_sq(vector<value_t> h_values) const {
    value_t result = 0.0;
    for (index_t ind = 0; ind < nnz; ++ind) {
      index_t i = indices[ind];
      result += values[ind] * values[ind] * h_values[i];
    }
    return result;
  }

  value_t ColumnFromPointers::mean() const {
    value_t sum = 0.0;
    for (index_t ind = 0; ind < nnz; ++ind)
      sum += values[ind];
    return sum / (value_t) length;
  }

  value_t ColumnFromPointers::l2_norm_sq() const {
    value_t result = 0.0;
    for (index_t ind = 0; ind < nnz; ++ind)
      result += values[ind] * values[ind];
    return result;
  }

  DatasetFromCSCPointers::DatasetFromCSCPointers(index_t *indices,
                                           nnz_t *indptr,
                                           value_t *data,
                                           value_t *labels,
                                           index_t n,
                                           index_t d,
                                           nnz_t nnz) {
    this->n = n;
    this->d = d;
    this->nnz = nnz;
    this->labels = labels;
    columns_vec.clear();
    columns_vec.reserve(d);
    for (index_t j = 0; j < d; ++j) {
      nnz_t offset = indptr[j];
      index_t col_nnz = (index_t) (indptr[j+1] - offset);
      value_t *col_data = data + offset;
      index_t *col_indices = indices + offset;
      Column *col = new ColumnFromPointers(col_indices, col_data, col_nnz, n);
      columns_vec.push_back(col);
    }
  }

  value_t DatasetFromCSCPointers::get_label(index_t i) const {
    return labels[i]; 
  }

  value_t l2_norm_sq(const vector<value_t> &vec) {
    value_t result = 0.0;
    for (size_t ind = 0; ind < vec.size(); ++ind)
      result += vec[ind] * vec[ind];
    return result;
  }

  value_t inner_product(const vector<value_t> &vec1, 
                        const vector<value_t> &vec2) {
    value_t result = 0.0;
    for (size_t ind = 0; ind < vec1.size(); ++ind)
      result += vec1[ind] * vec2[ind];
    return result;
  }

} // namespace BlitzL1
