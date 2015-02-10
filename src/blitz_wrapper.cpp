#include "../src/common.h"
#include <iostream>

using namespace BlitzL1;

extern "C" {

  Dataset* BlitzL1_new_sparse_dataset(index_t *indices,
                                      nnz_t *indptr,
                                      value_t *data,
                                      value_t *labels,
                                      index_t n,
                                      index_t d,
                                      nnz_t nnz) {
    return new DatasetFromCSCPointers(
                  indices, indptr, data, labels, n, d, nnz);
  }

  value_t BlitzL1_get_column_norm(Dataset* ds, index_t j) {
    return ds->get_column(j)->l2_norm();
  }

  value_t BlitzL1_get_label_i(Dataset* ds, index_t i) {
    return ds->get_label(i);
  }

}

