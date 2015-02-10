#include "common.h"
#include "solver.h"

#include <iostream>
using std::cout;
using std::endl;

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

  value_t BlitzL1_get_column_norm(Dataset* data, index_t j) {
    return data->get_column(j)->l2_norm();
  }

  value_t BlitzL1_get_label_i(Dataset* data, index_t i) {
    return data->get_label(i);
  }

  Solver* BlitzL1_new_solver() {
    return new Solver();
  }

  void BlitzL1_solve_problem(Solver *solver,
                           Dataset *data,
                           value_t lambda,
                           char *loss_type,
                           value_t *x,
                           value_t &intercept) {
    intercept = 2.5;
    solver->solve(data, lambda, loss_type, x, intercept, "");
  }
                           

}

