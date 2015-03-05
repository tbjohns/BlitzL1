#pragma once

#include <cstddef>
#include <vector>
#include <cmath>

namespace BlitzL1 {

  typedef double value_t; 
  typedef int index_t;

  class Column {
    protected:
      size_t length;
      size_t nnz;
      mutable value_t l2_norm_cache;
      mutable value_t l2_norm_centered_cache;
    
    public:
      Column() : l2_norm_cache(-1.0), l2_norm_centered_cache(-1.0) {}
      index_t get_length() const { return length; }
      index_t get_nnz() const { return nnz; } 
      virtual value_t inner_product(const std::vector<value_t> &vec) const = 0;
      virtual value_t weighted_inner_product(const std::vector<value_t> &vec, const std::vector<value_t> &weights) const = 0;
      virtual value_t h_norm_sq(const std::vector<value_t> &h_values) const = 0;
      virtual void add_multiple(std::vector<value_t> &target, 
                                value_t scaler) const = 0;
      virtual value_t mean() const = 0;
      virtual value_t l2_norm_sq() const = 0;
      value_t l2_norm() const;
      value_t l2_norm_centered() const;
  };

  class SparseColumnFromPointers : public Column {
    index_t *indices;
    value_t *values;

    public:
      SparseColumnFromPointers(index_t *indices, 
                         value_t *values,
                         size_t nnz,
                         size_t length);
      value_t inner_product(const std::vector<value_t> &vec) const;
      value_t weighted_inner_product(const std::vector<value_t> &vec, const std::vector<value_t> &weights) const;
      void add_multiple(std::vector<value_t> &target, 
                        value_t scaler) const;
      value_t h_norm_sq(const std::vector<value_t> &h_values) const;
      value_t mean() const;
      value_t l2_norm_sq() const;
  };

  class DenseColumnFromPointers : public Column {
    value_t *values;

    public:
      DenseColumnFromPointers(value_t *values, index_t length);
      value_t inner_product(const std::vector<value_t> &vec) const;
      value_t weighted_inner_product(const std::vector<value_t> &vec, const std::vector<value_t> &weights) const;
      void add_multiple(std::vector<value_t> &target, value_t scaler) const;
      value_t h_norm_sq(const std::vector<value_t> &h_values) const;
      value_t mean() const;
      value_t l2_norm_sq() const;
  };


  class Dataset {
    protected:
      std::vector<Column*> columns_vec;
      size_t n;   // # training examples
      size_t d;   // # features
      size_t nnz; // # nonzero entries
      value_t* labels;

    public:
      const Column* get_column(index_t j) const { return columns_vec[j]; }
      value_t get_label(index_t i) const;
      size_t get_nnz() const { return nnz; }
      size_t get_n() const { return n; }
      size_t get_d() const { return d; }
  };

  class DatasetFromCSCPointers : public Dataset {
    public:
      DatasetFromCSCPointers(index_t *indices,
                          index_t *indptr,
                          value_t *data,
                          value_t *labels,
                          size_t n,
                          size_t d,
                          size_t nnz);
  };

  class DatasetFromFContiguousPointer : public Dataset {
    public:
      DatasetFromFContiguousPointer(value_t *data, 
                                    value_t *labels, 
                                    size_t n, 
                                    size_t d);
  };

  value_t l2_norm_sq(const std::vector<value_t> &vec);
  value_t l2_norm_sq(const value_t *values, size_t length);
  value_t l1_norm(const value_t *x, size_t d);
  size_t l0_norm(const value_t *x, size_t d);
  value_t l2_norm_diff_sq(const std::vector<value_t> &vec1,
                          const std::vector<value_t> &vec2);
  value_t sum_vector(const std::vector<value_t> &vec);
  value_t sum_array(const value_t *values, size_t length);
  void add_scaler(std::vector<value_t> &vec, value_t scaler);
  value_t inner_product(const std::vector<value_t> &vec1, 
                        const std::vector<value_t> &vec2);
  value_t max_abs(const std::vector<value_t> &vec);
  value_t soft_threshold(value_t value, value_t threshold);
  void scale_vector(std::vector<value_t> &vec, value_t scale);
  void copy_and_scale_vector(const std::vector<value_t> &values, 
                             value_t scale,
                             std::vector<value_t> &target);

  class IndirectComparator {
    const std::vector<value_t>& values;
    IndirectComparator();

    public:
      IndirectComparator(const std::vector<value_t> &v) : values(v) {}
      bool operator() (const index_t &i, const index_t &j) {
        return values[i] < values[j];
      }
  };

  class IndirectExceedsThreshold {
    const std::vector<value_t>& values;
    value_t threshold;
    IndirectExceedsThreshold();

    public:
      IndirectExceedsThreshold(const std::vector<value_t> &v, value_t t) 
        : values(v), threshold(t) {}
      bool operator() (const size_t &j) {
        return (values[j] > threshold);
      }
  };

}
