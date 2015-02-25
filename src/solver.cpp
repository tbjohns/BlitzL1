#include "solver.h"
#include "loss.h"
#include "logger.h"
#include "timer.h"

#include <cstring>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <limits>

using namespace BlitzL1;

using std::vector;
using std::cout;
using std::endl;
using std::min;

const int MAX_BACKTRACK_ITR = 20;

void Solver::update_intercept(value_t &intercept, 
                              const Loss *loss_function,
                              const Dataset *data) {
  if (!use_intercept)
    return;

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

value_t Solver::compute_lambda_max(const Dataset *data, const char* loss_type) {

  Loss* loss_function = get_loss_function(loss_type);    
  index_t d = data->get_d();
  vector<value_t> x(d, 0.0);
  value_t intercept = 0.0;
  loss_function->compute_dual_points(
                    theta, aux_dual, &x[0], intercept, data);
  update_intercept(intercept, loss_function, data);
  update_ATtheta(data);
  return max_abs(ATtheta);
}

void Solver::run_prox_newton_iteration(value_t *x, 
                                       value_t &intercept, 
                                       value_t lambda,
                                       const Loss *loss_function, 
                                       const Dataset *data) {

  // Initialize iteration:
  index_t n = data->get_n();
  vector<value_t> Delta_x(working_set_size, 0.0);
  vector<value_t> A_Delta_x(n, 0.0);
  value_t Delta_intercept = 0.0;

  // Set up gradient and hessian values:
  vector<value_t> H;
  loss_function->compute_H(H, theta, aux_dual, data);
  vector<value_t> grad_cache(working_set_size, 0.0);
  vector<value_t> hess_cache(working_set_size, 0.0);
  for (index_t ind = 0; ind < working_set_size; ++ind) {
    index_t j = prioritized_features[ind];
    grad_cache[ind] = data->get_column(j)->inner_product(theta);
    hess_cache[ind] = data->get_column(j)->h_norm_sq(H);
  }

  // Cache values for updating intercept:
  value_t sum_theta = sum_vector(theta);
  value_t sum_H = sum_vector(H);
  
  // Approximately solve for newton direction:
  for (int cd_itr = 0; cd_itr < 10; ++cd_itr) {

    for (index_t ind = 0; ind < working_set_size; ++ind) {
      index_t j = prioritized_features[ind];
      const Column *col = data->get_column(j);

      // Compute hessian and gradient for coordinate j:
      value_t hess = hess_cache[ind];
      if (hess <= 0.0)
        continue;

      value_t grad = grad_cache[ind] 
                   + col->weighted_inner_product(A_Delta_x, H);

      // Apply coordinate descent update:
      value_t old_value = x[j] + Delta_x[ind]; 
      value_t proposal = old_value - grad / hess;
      value_t new_value = soft_threshold(proposal, lambda / hess);
      value_t diff = new_value - old_value;

      if (diff == 0.0)
        continue;

      Delta_x[ind] = new_value - x[j];
      col->add_multiple(A_Delta_x, diff);
    }

    // Update intercept:
    if (use_intercept) {
      value_t ip = inner_product(H, A_Delta_x);
      value_t diff = -(ip + sum_theta)/sum_H;
      Delta_intercept += diff;
      add_scaler(A_Delta_x, diff);
    }
  }

  // Apply update with backtracking:
  value_t t = 1.0;
  value_t last_t = 0.0;
  for (int backtrack_itr = 0; backtrack_itr < MAX_BACKTRACK_ITR; ++backtrack_itr) {
    value_t diff_t = t - last_t;

    intercept += diff_t * Delta_intercept;
    value_t subgrad_t = 0.0;

    for (index_t ind = 0; ind < working_set_size; ++ind) {
      index_t j = prioritized_features[ind];
      x[j] += diff_t * Delta_x[ind];
      if (x[j] < 0)
        subgrad_t -= lambda * Delta_x[ind];
      else if (x[j] > 0)
        subgrad_t += lambda * Delta_x[ind];
      else
        subgrad_t -= lambda * std::abs(Delta_x[ind]);
    }
    loss_function->compute_dual_points(
                      theta, aux_dual, x, intercept, data);
    subgrad_t += inner_product(A_Delta_x, theta);

    if (subgrad_t < 0) {
      break;
    } else {
      last_t = t;
      t *= 0.5;
    }
  }
}

value_t Solver::priority_norm_j(index_t j, const Dataset* data) {
  if (use_intercept)
    return data->get_column(j)->l2_norm_centered();
  else
    return data->get_column(j)->l2_norm();
}

value_t Solver::compute_alpha(const Dataset* data, value_t lambda, value_t theta_scale) {
  value_t best_alpha = 1.0;
  index_t d = data->get_d();
  for (index_t j = 0; j < d; ++j) {
    value_t norm = priority_norm_j(j, data);
    if (norm <= 0.0)
      continue;

    value_t l = ATphi[j];
    value_t r = theta_scale * ATtheta[j];
    if (std::abs(r) <= lambda)
      continue;

    value_t alpha;
    if (r >= 0)
      alpha = (lambda - l)/(r - l);
    else
      alpha = (-lambda - l)/(r - l);

    if (alpha < best_alpha)
      best_alpha = alpha;
  }
  return best_alpha;
}

void Solver::update_ATtheta(const Dataset *data) {
  index_t d = data->get_d();
  ATtheta.resize(d);
  for (index_t j = 0; j < d; ++j) {
    ATtheta[j] = data->get_column(j)->inner_product(theta);
  }
}

void Solver::update_phi(value_t alpha, value_t theta_scale) {
  for (size_t j = 0; j < ATphi.size(); ++j) 
    ATphi[j] = (1 - alpha) * ATphi[j] + alpha * theta_scale * ATtheta[j];
  for (size_t i = 0; i < phi.size(); ++i)
    phi[i] = (1 - alpha) * phi[i] + alpha * theta_scale * theta[i];
}

void Solver::prioritize_features(const Dataset *data, value_t lambda, const value_t *x) {
  index_t d = data->get_d();
  feature_priorities.resize(d);
  for (index_t j = 0; j < d; ++j) {
    if (x[j] != 0) {
      feature_priorities[j] = 0.0;
    } else {
      value_t AjTphi = ATphi[j];
      value_t norm = priority_norm_j(j, data);
      if (norm <= 0) {
        feature_priorities[j] = std::numeric_limits<value_t>::max();
      } else {
        value_t priority_value = (lambda - std::abs(AjTphi)) / norm;
        feature_priorities[j] = priority_value;
      }
    }
  }
  IndirectComparator cmp(feature_priorities);
  sort(
    prioritized_features.begin(),
    prioritized_features.end(),
    cmp);
}

void Solver::solve(const Dataset *data,
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

  num_iterations = 0;
  Loss* loss_function = get_loss_function(loss_type);
  index_t n = data->get_n();
  index_t d = data->get_d();
  working_set_size = 0;
  prioritized_features.resize(d);
  for (index_t j = 0; j < d; ++j)
    prioritized_features[j] = j;

  // Initialize dual points:
  loss_function->compute_dual_points(
                    theta, aux_dual, x, intercept, data);
  phi.assign(n, 0.0);
  ATphi.assign(d, 0.0);
  ATtheta.assign(d, 0.0);

  // Initialize objective values:
  value_t primal_loss = loss_function->primal_loss(theta, aux_dual);
  primal_obj = primal_loss + lambda * l1_norm(x, d);

  // Main Blitz loop:
  while (++num_iterations) {
    value_t primal_obj_last = primal_obj;

    update_intercept(intercept, loss_function, data);

    update_ATtheta(data);

    // Compute theta scale:
    value_t max_grad_working_set = 0.0;
    for (index_t ind = 0; ind < working_set_size; ++ind) {
      index_t j = prioritized_features[ind]; 
      value_t AjTtheta = ATtheta[j];
      if (std::abs(AjTtheta) > max_grad_working_set)
        max_grad_working_set = std::abs(AjTtheta);
    }
    value_t theta_scale = 1.0;
    if (max_grad_working_set > lambda)
      theta_scale = lambda / max_grad_working_set;

    value_t alpha = compute_alpha(data, lambda, theta_scale);
    update_phi(alpha, theta_scale);

    value_t dual_obj = loss_function->dual_obj(phi, data);

    prioritize_features(data, lambda, x);

    // Determine working set size:
    working_set_size = 2 * l0_norm(x, d);
    if (working_set_size < 100)
      working_set_size = 100;
    if (working_set_size > d)
      working_set_size = d;
  
    // Solve subproblem:
    for (int prox_newt_itr = 0; prox_newt_itr < 5; prox_newt_itr++){
      run_prox_newton_iteration(x, intercept, lambda, loss_function, data); 
      update_intercept(intercept, loss_function, data);
    }

    primal_loss = loss_function->primal_loss(theta, aux_dual);
    primal_obj = primal_loss + lambda * l1_norm(x, d); 
    duality_gap = primal_obj - dual_obj;

    // Print/record results:
    double elapsed_time = timer.elapsed_time();
    timer.pause_timing();
    if (verbose)
      cout << "Time: " << elapsed_time
           << " Objective: " << primal_obj 
           << " Duality gap: " << duality_gap << endl;

    logger.log_point(elapsed_time, primal_obj);
    timer.continue_timing();

    // Test stopping conditions:
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

