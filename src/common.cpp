#include "common.h"

using std::vector;

#include <iostream>
using std::cout;
using std::endl;

namespace BlitzL1 {

  value_t Column::l2_norm_centered() const {
    value_t value_sq = l2_norm_sq() - mean() * mean() * length;
    if (value_sq <= 0)
      return 0;
    else
      return sqrt(value_sq);
  }

  SparseColumnFromPointers::SparseColumnFromPointers(
      index_t *indices, value_t *values, index_t nnz, index_t length) {
    this->indices = indices;
    this->values = values;
    this->nnz = nnz;
    this->length = length;
  }

  value_t SparseColumnFromPointers::inner_product(const vector<value_t> &vec) const {
    value_t result = 0.0;
    for (index_t ind = 0; ind < nnz; ++ind) {
      index_t i = indices[ind];
      result += values[ind] * vec[i];
    }
    return result;
  }

  void SparseColumnFromPointers::add_multiple(vector<value_t> &target, value_t scaler) const {
    for (index_t ind = 0; ind < nnz; ++ind) {
      index_t i = indices[ind];
      target[i] += values[ind] * scaler;
    }
  }

  value_t SparseColumnFromPointers::h_norm_sq(const vector<value_t> &h_values) const {
    value_t result = 0.0;
    for (index_t ind = 0; ind < nnz; ++ind) {
      index_t i = indices[ind];
      result += values[ind] * values[ind] * h_values[i];
    }
    return result;
  }

  value_t SparseColumnFromPointers::mean() const {
    value_t sum = sum_array(values, nnz);
    return sum / (value_t) length;
  }

  value_t SparseColumnFromPointers::l2_norm_sq() const {
    return BlitzL1::l2_norm_sq(values, nnz);
  }

  
  DenseColumnFromPointers::DenseColumnFromPointers(
          value_t *values, index_t length) {
    this->values = values;
    this->nnz = length;
    this->length = length;
  }

  value_t DenseColumnFromPointers::inner_product(const vector<value_t> &vec) const {
    value_t result = 0.0;
    for (index_t i = 0; i < length; ++i)
      result += values[i] * vec[i];
    return result;
  }


  void DenseColumnFromPointers::add_multiple(vector<value_t> &target, value_t scaler) const {
    for (index_t i = 0; i < length; ++i)
      target[i] += values[i] * scaler;
  }

  value_t DenseColumnFromPointers::h_norm_sq(const vector<value_t> &h_values) const {
    value_t result = 0.0;
    for (index_t i = 0; i < length; ++i)
      result += values[i] * values[i] * h_values[i];
    return result;
  }

  value_t DenseColumnFromPointers::mean() const {
    value_t sum = sum_array(values, length);
    return sum / length;
  }

  value_t DenseColumnFromPointers::l2_norm_sq() const {
    return BlitzL1::l2_norm_sq(values, length);
  }

  value_t Dataset::get_label(index_t i) const {
    return labels[i]; 
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
      Column *col = new SparseColumnFromPointers(col_indices, col_data, col_nnz, n);
      columns_vec.push_back(col);
    }
  }

  DatasetFromFContiguousPointer::DatasetFromFContiguousPointer(
        value_t *data, value_t *labels, index_t n, index_t d) {
    this->n = n;
    this->d = d;
    this->nnz = n * d;
    this->labels = labels;
    columns_vec.clear();
    columns_vec.reserve(d);
    for (index_t j = 0; j < d; ++j) {
      value_t *col_data = data + j * n; 
      Column *col = new DenseColumnFromPointers(col_data, n);
      columns_vec.push_back(col);
    }
  }


  value_t l2_norm_sq(const vector<value_t> &vec) {
    value_t result = 0.0;
    for (size_t ind = 0; ind < vec.size(); ++ind)
      result += vec[ind] * vec[ind];
    return result;
  }

  value_t l2_norm_sq(const value_t *values, index_t length) {
    value_t result =  0.0;
    for (index_t ind = 0; ind < length; ++ind)
      result += values[ind] * values[ind];
    return result;
  }

  value_t l1_norm(value_t *vec, index_t len) {
    value_t result = 0.0;
    for (index_t ind = 0; ind < len; ++ind)
      result += std::abs(vec[ind]);
    return result;
  }

  value_t sum_array(const value_t *values, index_t length) {
    value_t result =  0.0;
    for (index_t ind = 0; ind < length; ++ind)
      result += values[ind];
    return result;
  }

  value_t sum_vector(const vector<value_t> &vec) {
    value_t result = 0.0;
    for (size_t ind = 0; ind < vec.size(); ++ind)
      result += vec[ind];
    return result;
  }

  value_t inner_product(const vector<value_t> &vec1, 
                        const vector<value_t> &vec2) {
    value_t result = 0.0;
    for (size_t ind = 0; ind < vec1.size(); ++ind)
      result += vec1[ind] * vec2[ind];
    return result;
  }

  value_t max_abs(const vector<value_t> &vec) {
    value_t result = 0.0;
    for (size_t ind = 0; ind < vec.size(); ++ind) {
      if (std::abs(vec[ind]) > result)
        result = std::abs(vec[ind]);
    }
    return result;
  }

  value_t soft_threshold(value_t value, value_t threshold) {
    if (value > threshold)
      return value - threshold;
    else if (value < -threshold)
      return value + threshold;
    else
      return 0.0;
  }

  void scale_vector(vector<value_t> &vec, value_t scale) {
    for (size_t ind = 0; ind < vec.size(); ++ind)
      vec[ind] *= scale;
  }

}
