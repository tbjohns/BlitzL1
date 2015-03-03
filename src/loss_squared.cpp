#include "loss.h"

using namespace BlitzL1;

using std::vector;

SquaredLoss::SquaredLoss() : L(1.0) {}

value_t SquaredLoss::primal_loss(const vector<value_t> &theta,
                                 const vector<value_t> &aux_dual,
                                 const Dataset *data) const {
  return 0.5 * l2_norm_sq(theta);
}

value_t SquaredLoss::dual_obj(const vector<value_t> &theta,
                              const Dataset *data,
                              value_t theta_scaler) const {
  value_t dot_prod = 0;
  index_t n = data->get_n();
  for (index_t i = 0; i < n; ++i)
    dot_prod += data->get_label(i) * theta[i];
  return -0.5 * theta_scaler * theta_scaler * l2_norm_sq(theta) 
         - theta_scaler * dot_prod;
}

void SquaredLoss::compute_dual_points(
                              vector<value_t> &Ax,
                              vector<value_t> &theta,
                              vector<value_t> &aux_dual,
                              const Dataset* data) const {
  size_t n = data->get_n();
  theta.resize(n);
  for (size_t i = 0; i < n; ++i) {
    theta[i] = Ax[i] - data->get_label(i);
  }
}

void SquaredLoss::compute_H(vector<value_t> &H,
               const vector<value_t> &theta,
               const vector<value_t> &Ax,
               const Dataset* data) const {
  index_t n = data->get_n();
  H.assign(n, 1.0);
}

void SquaredLoss::apply_intercept_update(
                value_t delta,
                vector<value_t> &Ax, 
                vector<value_t> &theta, 
                vector<value_t> &aux_dual,
                const Dataset* data) const {
  index_t n = data->get_n(); 
  for (index_t i = 0; i < n; ++i) {
    Ax[i] += delta;
    theta[i] += delta;
  }
}


