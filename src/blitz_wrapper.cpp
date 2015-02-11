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

  Dataset* BlitzL1_new_dense_dataset(value_t *data,
                                     value_t *labels,
                                     index_t n,
                                     index_t d) {
    cout << "data[0] is " << *data << endl;
    return new DatasetFromFContiguousPointer(data, labels, n, d);
  }

  void BlitzL1_free_dataset(Dataset* data) {
    delete data;
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

  void BlitzL1_set_tolerance(Solver *solver, value_t value) {
    solver->set_tolerance(value);
  }

  value_t BlitzL1_get_tolerance(Solver *solver) {
    return solver->get_tolerance();
  }

  void BlitzL1_set_max_time(Solver *solver, value_t value) {
    solver->set_max_time(value);
  }

  value_t BlitzL1_get_max_time(Solver *solver) {
    return solver->get_max_time();
  }

  void BlitzL1_set_min_time(Solver *solver, value_t value) {
    solver->set_min_time(value);
  }

  value_t BlitzL1_get_min_time(Solver *solver) {
    return solver->get_min_time();
  }

  void BlitzL1_set_use_intercept(Solver *solver, bool value) {
    solver->set_use_intercept(value);
  }

  bool BlitzL1_get_use_intercept(Solver *solver) {
    return solver->get_use_intercept();
  }

  void BlitzL1_set_verbose(Solver *solver, bool value) {
    solver->set_verbose(value);
  }

  bool BlitzL1_get_verbose(Solver *solver) {
    return solver->get_verbose();
  }

  void BlitzL1_solve_problem(Solver *solver,
                           Dataset *data,
                           value_t lambda,
                           char *loss_type,
                           value_t *x,
                           value_t &intercept,
                           value_t &primal_obj,
                           value_t &duality_gap,
                           char *log_dir) {
    solver->solve(data, lambda, loss_type, x, intercept, primal_obj, duality_gap, log_dir);
  }

                           

}

