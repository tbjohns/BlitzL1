#pragma once

#include <vector>
#include <cmath>

namespace BlitzL1 {

  typedef double value_t; 
  typedef int index_t;
  typedef int nnz_t;
  typedef std::size_t size_t;

  class Column {
    protected:
      index_t length;
      index_t nnz;
    
    public:
      index_t get_length() const { return length; }
      index_t get_nnz() const { return nnz; } 
      virtual value_t inner_product(const std::vector<value_t> &vec) const = 0;
      virtual value_t h_norm_sq(const std::vector<value_t> &h_values) const = 0;
      virtual void add_multiple(std::vector<value_t> &target, 
                                value_t scaler) const = 0;
      virtual value_t mean() const = 0;
      virtual value_t l2_norm_sq() const = 0;
      value_t l2_norm() const { return sqrt(l2_norm_sq()); }
      value_t l2_norm_centered() const;
  };

  class SparseColumnFromPointers : public Column {
    index_t *indices;
    value_t *values;

    public:
      SparseColumnFromPointers(index_t *indices, 
                         value_t *values,
                         index_t nnz,
                         index_t length);
      value_t inner_product(const std::vector<value_t> &vec) const;
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
      void add_multiple(std::vector<value_t> &target, value_t scaler) const;
      value_t h_norm_sq(const std::vector<value_t> &h_values) const;
      value_t mean() const;
      value_t l2_norm_sq() const;
  };


  class Dataset {
    protected:
      std::vector<Column*> columns_vec;
      index_t n;   // # training examples
      index_t d;   // # features
      index_t nnz; // # nonzero entries
      value_t* labels;

    public:
      const Column* get_column(index_t j) const { return columns_vec[j]; }
      value_t get_label(index_t i) const;
      index_t get_nnz() const { return nnz; }
      index_t get_n() const { return n; }
      index_t get_d() const { return d; }
  };

  class DatasetFromCSCPointers : public Dataset {
    public:
      DatasetFromCSCPointers(index_t *indices,
                          nnz_t *indptr,
                          value_t *data,
                          value_t *labels,
                          index_t n,
                          index_t d,
                          nnz_t nnz);
  };

  class DatasetFromFContiguousPointer : public Dataset {
    public:
      DatasetFromFContiguousPointer(value_t *data, 
                                    value_t *labels, 
                                    index_t n, 
                                    index_t d);
  };

  value_t l2_norm_sq(const std::vector<value_t> &vec);
  value_t l2_norm_sq(const value_t *values, index_t length);
  value_t sum_array(const value_t *values, index_t length);
  value_t l1_norm(value_t *x, index_t d);
  value_t inner_product(const std::vector<value_t> &vec1, 
                        const std::vector<value_t> &vec2);
  value_t max_abs(const std::vector<value_t> &vec);
  value_t soft_threshold(value_t value, value_t threshold);
}
