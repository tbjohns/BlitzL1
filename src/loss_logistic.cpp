#include "loss.h"
#include <math.h>
#include <limits>

using std::vector;
using std::abs;

#include <iostream>
using std::endl;
using std::cout;

using namespace BlitzL1;

const value_t MIN_ABS_LABEL = 1e-5;

LogisticLoss::LogisticLoss() : L(0.25) {}

value_t LogisticLoss::primal_loss(const vector<value_t> &theta,
                                 const vector<value_t> &aux_dual) const {
  value_t loss = 0.0;
  for (size_t i = 0; i < aux_dual.size(); ++i) {
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
    if (abs(label) < MIN_ABS_LABEL) {
      result += log(2.0);
    } else {
      value_t val = theta_scaler * theta[i] / label;
      result += val * log(-val) - (1 + val) * log1p(val);
    }
  }
  return result;
}

void LogisticLoss::compute_dual_points(
                              vector<value_t> &theta,
                              vector<value_t> &aux_dual,
                              const value_t *x,
                              value_t intercept,
                              const Dataset* data) const {
  compute_Ax(x, intercept, data, theta); // theta is now Ax
  index_t n = data->get_n();
  aux_dual.resize(n);
  for (index_t i = 0; i < n; ++i) {
    value_t minus_label = -data->get_label(i);
    aux_dual[i] = exp(minus_label * theta[i]);
    if (aux_dual[i] == std::numeric_limits<value_t>::infinity()) {
      cout << minus_label << endl;
      cout << theta[i] << endl;
    }
    theta[i] = minus_label * aux_dual[i] / (1.0 + aux_dual[i]);
    if (theta[i] != theta[i]) {
      cout << "died at " << i << endl;
      cout << "aux dual is " << aux_dual[i] << endl;
      cout << "label is " << minus_label << endl;
      throw i;
    }
    // theta[i] is now dual point
  }
}

void LogisticLoss::compute_H(vector<value_t> &H,
               const vector<value_t> &theta,
               const vector<value_t> &aux_dual,
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
                vector<value_t> &theta, 
                vector<value_t> &aux_dual, 
                const Dataset* data) const {
  index_t n = data->get_n(); 
  for (index_t i = 0; i < n; ++i) {
    value_t minus_label = -data->get_label(i);
    aux_dual[i] *= exp(minus_label * delta);
    if (aux_dual[i] == std::numeric_limits<value_t>::infinity())
      theta[i] = minus_label;
    else
      theta[i] = minus_label * aux_dual[i] / (1.0 + aux_dual[i]);
  }
}

