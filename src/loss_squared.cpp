#include "loss.h"

using std::vector;

using namespace BlitzL1;

SquaredLoss::SquaredLoss() : L(1.0) {}

value_t SquaredLoss::primal_loss(const vector<value_t> &theta,
                                 const vector<value_t> &aux_dual) {
  return 0.5 * l2_norm_sq(theta);
}

value_t SquaredLoss::dual_obj(const vector<value_t> &theta,
                              Dataset *data) {
  value_t dot_prod = 0;
  index_t n = data->get_n();
  for (index_t i = 0; i < n; ++i)
    dot_prod += data->get_label(i) * theta[i];
  return -0.5 * l2_norm_sq(theta) - dot_prod;
}

void SquaredLoss::compute_dual_points(
                              vector<value_t> &theta,
                              vector<value_t> &aux_dual,
                              const value_t *x,
                              value_t intercept,
                              Dataset* data) {
  compute_Ax(x, intercept, data, theta);
  index_t n = data->get_n();
  for (index_t i = 0; i < n; ++i) {
    theta[i] -= data->get_label(i);
  }
}

void SquaredLoss::compute_H(vector<value_t> &H,
               const vector<value_t> &theta,
               const vector<value_t> &aux_dual,
               Dataset* data) {
  index_t n = data->get_n();
  H.assign(n, 1.0);
}

void SquaredLoss::apply_intercept_update(
                value_t delta,
                vector<value_t> &theta, 
                vector<value_t> &aux_dual, 
                Dataset* data) {
  index_t n = data->get_n(); 
  for (index_t i = 0; i < n; ++i) 
    theta[i] += delta;
}


