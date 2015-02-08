#pragma once

#include <vector>
#include <math.h>

namespace BlitzL1 {

  typedef unsigned int index_t;
  typedef double value_t; 

  class Column {
    protected:
      index_t length;
      index_t nnz;
    
    public:
      index_t get_length() const { return length; }
      index_t get_nnz() const { return nnz; } 
      virtual value_t inner_product(std::vector<value_t> vec) const = 0;
      virtual value_t h_norm_sq(std::vector<value_t> h_values) const = 0;
      virtual value_t mean() const = 0;
      virtual value_t l2_norm_sq() const = 0;
      value_t l2_norm() { return sqrt(l2_norm_sq()); }
      value_t l2_norm_centered();
  };

  class ColumnFromPointers : public Column {
    index_t *indices;
    value_t *values;

    public:
      ColumnFromPointers(index_t *indices, 
                         value_t *values,
                         index_t nnz,
                         index_t length);
      value_t inner_product(std::vector<value_t> vec) const;
      value_t h_norm_sq(std::vector<value_t> h_values) const;
      value_t mean() const;
      value_t l2_norm_sq() const;
  };

  class Dataset {

      value_t *labels;
      std::vector<Column*> columns_vec;
      index_t n;
      index_t d;
      index_t nnz;

      const Column* get_column(index_t j) const { return columns_vec[j]; }
      value_t get_label(index_t i) const { return labels[i]; }

  };

  struct ProblemOptions {
    double max_time;  
    double min_time;
    value_t epsilon;
    bool verbose;
  };

}
