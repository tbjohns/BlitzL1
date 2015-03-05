#include "common.h"
#include "solver.h"

using namespace BlitzL1;

extern "C" {

  Dataset* BlitzL1_new_sparse_dataset(index_t *indices,
                                      index_t *indptr,
                                      value_t *data,
                                      value_t *labels,
                                      size_t n,
                                      size_t d,
                                      size_t nnz) {
    return new DatasetFromCSCPointers(
                  indices, indptr, data, labels, n, d, nnz);
  }

  Dataset* BlitzL1_new_dense_dataset(value_t *data,
                                     value_t *labels,
                                     size_t n,
                                     size_t d) {
    return new DatasetFromFContiguousPointer(data, labels, n, d);
  }

  value_t BlitzL1_compute_lambda_max(Solver *solver, const Dataset* data, char* loss_type) {
    return solver->compute_lambda_max(data, loss_type);
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
                             const Dataset *data,
                             value_t lambda,
                             const char *loss_type,
                             value_t *x,
                             value_t &intercept,
                             char* solution_status,
                             value_t &primal_obj,
                             value_t &duality_gap,
                             int &num_iterations,
                             char *log_dir) {
    solver->solve(data, lambda, loss_type, x, intercept, solution_status, 
                  primal_obj, duality_gap, num_iterations, log_dir);
  }

}

