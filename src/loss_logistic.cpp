#include "loss.h"
#include <math.h>
#include <limits>

using std::vector;

using namespace BlitzL1;

const value_t MIN_ABS_LABEL = 1e-5;

LogisticLoss::LogisticLoss() : L(0.25) {}

value_t LogisticLoss::primal_loss(const vector<value_t> &theta,
                                  const vector<value_t> &aux_dual,
                                  const Dataset *data) const {
  value_t loss = 0.0;
  size_t n = data->get_n();
  for (size_t i = 0; i < n; ++i) {
    loss += log1p(aux_dual[i]);
  }
  return loss;
}

value_t LogisticLoss::dual_obj(const vector<value_t> &theta,
                               const Dataset *data,
                               value_t theta_scaler) const {
  value_t result = 0.0;
  index_t n = data->get_n();
  for (index_t i = 0; i < n; ++i) {
    value_t label = data->get_label(i);
    if (fabs(label) < MIN_ABS_LABEL) {
      result += log(2.0);
    } else {
      value_t val = theta_scaler * theta[i] / label;
      result += val * log(-val) - (1 + val) * log1p(val);
    }
  }
  return result;
}

void LogisticLoss::compute_dual_points(
                              vector<value_t> &Ax,
                              vector<value_t> &theta,
                              vector<value_t> &aux_dual,
                              const Dataset* data) const {

  index_t n = data->get_n();
  theta.resize(n);
  aux_dual.resize(n);
  for (index_t i = 0; i < n; ++i) {
    value_t minus_label = -data->get_label(i);
    aux_dual[i] = exp(minus_label * Ax[i]);
    theta[i] = minus_label * aux_dual[i] / (1.0 + aux_dual[i]);
  }
}

void LogisticLoss::compute_H(vector<value_t> &H,
               const vector<value_t> &theta,
               const vector<value_t> &Ax,
               const Dataset* data) const {
  index_t n = data->get_n();
  H.resize(n);
  for (index_t i = 0; i < n; ++i) {
    value_t label = data->get_label(i);
    H[i] = -theta[i] * (label + theta[i]);
  }
}

void LogisticLoss::apply_intercept_update(
                value_t delta,
                vector<value_t> &Ax, 
                vector<value_t> &theta, 
                vector<value_t> &aux_dual,
                const Dataset* data) const {
  index_t n = data->get_n(); 
  for (index_t i = 0; i < n; ++i) {
    value_t minus_label = -data->get_label(i);
    Ax[i] += delta;
    aux_dual[i] = exp(minus_label * Ax[i]);
    theta[i] = minus_label * aux_dual[i] / (1.0 + aux_dual[i]);
  }
}

