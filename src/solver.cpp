#include "solver.h"
#include "loss.h"
#include "logger.h"
#include "timer.h"

#include <cstring>
#include <stdio.h>
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

Loss* get_loss_function(const char* loss_type) {
  if (strcmp(loss_type, "logistic") == 0)
    return new LogisticLoss();
  else if (strcmp(loss_type, "squared") == 0)
    return new SquaredLoss();
  else
    throw loss_type;
}

value_t Solver::compute_lambda_max(Dataset *data, const char* loss_type) {

  Loss* loss_function = get_loss_function(loss_type);    
  index_t d = data->get_d();
  vector<value_t> x(d, 0.0);
  value_t intercept = 0.0;
  loss_function->compute_dual_points(
                    theta, aux_dual, &x[0], intercept, data);
  if (use_intercept)
    update_intercept(intercept, loss_function, data);

  value_t lambda_max = 0.0;
  for (index_t j = 0; j < d; ++j) {
    value_t ip = data->get_column(j)->inner_product(theta);
    value_t val = std::abs(ip);
    if (val > lambda_max)
      lambda_max = val;
  }

  return lambda_max;
}

/*
void Solver::run_prox_newton_iteration() {
  // Data
  // loss_function

  // Initialize iteration:
  index_t n = data->get_n();
  index_t d = data->get_d();
  vector<value_t> Delta_x(d, 0.0);
  vector<value_t> A_Delta_x.assign(n, 0.0);
  value_t Delta_intercept = 0.0;

  // Set up gradient and hessian values:
  vector<value_t> hess_cache;
  loss_function->compute_H(hess_cache, theta, aux_dual, data);
  vector<value_t> grad_cache(d, 0.0);
  for (index_t j = 0; j < d; ++j)
    grad_cache[j] = data->get_column(j)->ip(theta);
  
  // Approximately solve for newton direction:
  for (int cd_itr = 0; cd_itr < 15; ++cd_itr) {

    for (index_t j = 0; j < d; ++j) {
      const Column *col = data->get_column(j);

      value_t hess = hess_cache[j];
      value_t grad = grad_cache[j];
    }

  }
}
*/

void Solver::solve(Dataset *data,
                   value_t lambda,
                   const char *loss_type,
                   value_t* x,
                   value_t &intercept,
                   char *solution_status,
                   value_t &primal_obj,
                   value_t &duality_gap,
                   int &num_iterations,
                   const char *log_directory) {

  Timer timer;
  Logger logger(log_directory);

  Loss* loss_function = get_loss_function(loss_type);
  loss_function->compute_dual_points(
                    theta, aux_dual, x, intercept, data);

  num_iterations = 0;
  index_t d = data->get_d();
  value_t primal_loss = loss_function->primal_loss(theta, aux_dual);
  primal_obj = primal_loss + lambda * l1_norm(x, d);
  value_t L = 5 * loss_function->L;

  while (true) {
    num_iterations++;

    if (use_intercept)
      update_intercept(intercept, loss_function, data);

    value_t primal_loss_last = primal_loss;
    value_t primal_obj_last = primal_obj;
    vector<value_t> loss_gradients;
    loss_gradients.assign(d, 0.0);
    for (index_t j = 0; j < d; ++j) {
      loss_gradients[j] = data->get_column(j)->inner_product(theta);
    }

    value_t max_loss_gradient = max_abs(loss_gradients);
    vector<value_t> omega(theta);
    scale_vector(omega, lambda / max_loss_gradient);
    value_t dual_obj = loss_function->dual_obj(omega, data);

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
    duality_gap = primal_obj - dual_obj;

    double elapsed_time = timer.elapsed_time();
    timer.pause_timing();
    if (verbose)
      cout << "Time: " << elapsed_time
           << " Objective: " << primal_obj 
           << " Duality gap: " << duality_gap << endl;

    logger.log_point(elapsed_time, primal_obj);
    timer.continue_timing();

    if ((duality_gap / std::abs(dual_obj) < tolerance) &&
        (elapsed_time > min_time)) {
      sprintf(solution_status, "reached stopping tolerance");
      break;
    }

    if (primal_obj >= primal_obj_last) {
      sprintf(solution_status, "reached machine precision");
      break;
    }

    if (elapsed_time > max_time) {
      sprintf(solution_status, "reached time limit");
      break;
    }

  }

}

