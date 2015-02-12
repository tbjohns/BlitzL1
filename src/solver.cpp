#include "solver.h"
#include "loss.h"

#include <cstring>
#include <iostream>

using namespace BlitzL1;

using std::vector;
using std::cout;
using std::endl;

const int MAX_BACKTRACK = 20;
const value_t L_INCREASE_RATIO = 1.25;

void Solver::update_intercept(value_t &intercept, 
                              Loss *loss_function,
                              Dataset *data) {
  // 1-d newton method:
  vector<value_t> H;
  for (int itr = 0; itr < 10; ++itr) {
    value_t grad = sum_vector(theta);
    loss_function->compute_H(H, theta, aux_dual, data);
    value_t hess = sum_vector(H);
    value_t delta = -grad / hess;
    intercept += delta;
    loss_function->apply_intercept_update(delta, theta, aux_dual, data);

    if (std::abs(grad) <= 1e-14)
      break;
  }
}

void Solver::solve(Dataset *data,
                   value_t lambda,
                   char* loss_type,
                   value_t* x,
                   value_t &intercept,
                   value_t &primal_obj,
                   value_t &duality_gap,
                   char* log_directory) {

  Loss* loss_function;
  if (strcmp(loss_type, "logistic") == 0)
    loss_function = new LogisticLoss();
  else if (strcmp(loss_type, "squared") == 0)
    loss_function = new SquaredLoss();
  else
    throw loss_type;

  index_t d = data->get_d();

  loss_function->compute_dual_points(
                    theta, aux_dual, x, intercept, data);
  value_t primal_loss = loss_function->primal_loss(theta, aux_dual);
  primal_obj = primal_loss + lambda * l1_norm(x, d);
  value_t L = 5 * loss_function->L;

  while (true) {
    if (use_intercept)
      update_intercept(intercept, loss_function, data);

    value_t primal_loss_last = primal_loss;
    value_t primal_obj_last = primal_obj;
    vector<value_t> loss_gradients;
    loss_gradients.assign(d, 0.0);
    for (index_t j = 0; j < d; ++j) {
      loss_gradients[j] = data->get_column(j)->inner_product(theta);
    }

    vector<value_t> new_x(d, 0.0);
    int backtrack_itr = 0;
    while (backtrack_itr++ < MAX_BACKTRACK) {
      for (index_t j = 0; j < d; ++j) {
        value_t norm_sq = data->get_column(j)->l2_norm_sq();
        value_t new_value = x[j] - loss_gradients[j] / norm_sq / L;
        value_t threshold = lambda / norm_sq / L;
        new_x[j] = soft_threshold(new_value, threshold);
      }
      
      loss_function->compute_dual_points(
                        theta, aux_dual, &new_x[0], intercept, data);
      primal_loss = loss_function->primal_loss(theta, aux_dual);

      value_t Q = primal_loss_last;
      for (index_t j = 0; j < d; ++j) {
        value_t norm_sq = data->get_column(j)->l2_norm_sq();
        value_t delta = new_x[j] - x[j];
        Q += delta * loss_gradients[j];
        Q += L / 2 * norm_sq * delta * delta;
      }

      if (primal_loss <= Q) {
        break;
      } else {
        L *= L_INCREASE_RATIO;
      }
    }
    
    for (index_t j = 0; j < d; ++j)
      x[j] = new_x[j];

    primal_obj = primal_loss + lambda * l1_norm(x, d); 
    if (verbose)
      cout << "Objective: " << primal_obj << endl;

    if (primal_obj >= primal_obj_last) {
      break;
    }

  }

}

