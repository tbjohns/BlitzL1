#include "solver.h"
#include "loss.h"
#include "logger.h"
#include "timer.h"

#include <cstring>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <limits>

using std::vector;
using std::cout;
using std::endl;
using std::min;
using std::swap;

namespace BlitzL1 {

const int MAX_BACKTRACK_ITR = 20;
const int MAX_PROX_NEWTON_CD_ITR = 20;
const value_t PROX_NEWTON_EPSILON_RATIO = 10.0;
const int MIN_PROX_NEWTON_CD_ITR = 2;

value_t Solver::update_intercept(value_t &intercept, 
                                 const Loss *loss_function,
                                 const Dataset *data) {
  // Optimizes over intercept variable with other weights fixed
  // Modifies intercept, theta, and Ax
  // Returns change in intercept value

  if (!use_intercept)
    return 0.0;

  // 1-d newton method:
  value_t intercept_old = intercept;
  vector<value_t> H;
  for (int itr = 0; itr < 10; ++itr) {
    value_t grad = sum_vector(theta);
    if (fabs(grad) <= 1e-14)
      break;
    loss_function->compute_H(H, theta, Ax, data);
    value_t hess = sum_vector(H);
    value_t delta = -grad / hess;
    loss_function->apply_intercept_update(
                                delta, Ax, theta, aux_dual, data);
    intercept += delta;
  }

  return intercept - intercept_old;
}

Loss* get_loss_function(const char* loss_type) {
  // Returns pointer to new loss object of type "loss_type"
  // Raises exception if loss type not recognized

  if (strcmp(loss_type, "logistic") == 0)
    return new LogisticLoss();
  else if (strcmp(loss_type, "squared") == 0)
    return new SquaredLoss();
  else
    throw loss_type;
}

value_t Solver::compute_lambda_max(const Dataset *data, const char* loss_type) {
  // Returns smallest lambda value for which solution is entirely 0

  // Initialize theta:
  Loss* loss_function = get_loss_function(loss_type);    
  size_t n = data->get_n();
  size_t d = data->get_d();
  Ax.assign(n, 0.0);
  loss_function->compute_dual_points(Ax, theta, aux_dual, data);

  // Center theta (if using intercept term):
  value_t intercept;
  update_intercept(intercept, loss_function, data);

  // Compute loss gradient for all features:
  prioritized_features.resize(d);
  for (size_t j = 0; j < d; ++j)
    prioritized_features[j] = j;
  update_ATtheta(data);

  // Return largest magnitude in loss gradients:
  return max_abs(ATtheta);
}

void Solver::reset_prox_newton_variables() {
  // Resets subproblem solver variables before solving subproblem

  first_prox_newton_iteration = true;
  prox_newton_grad_diff = 0.0;
}

value_t Solver::run_prox_newton_iteration(value_t *x, 
                                       value_t &intercept, 
                                       value_t lambda,
                                       const Loss *loss_function, 
                                       const Dataset *data) {
  // Runs one iteration of subproblem solver
  // Returns theta_scale, which is the largest scaler value <= 1
  // for which scaler*theta is feasible for the subproblem

  // Initialize iteration:
  size_t n = data->get_n();
  vector<value_t> Delta_x(working_set_size, 0.0);
  vector<value_t> A_Delta_x(n, 0.0);
  value_t Delta_intercept = 0.0;

  // Permutation vector for shuffling indices:
  vector<size_t> rand_permutation;
  rand_permutation.reserve(working_set_size);
  for (size_t ind = 0; ind < working_set_size; ++ind)
    rand_permutation.push_back(ind);

  // Set up hessian values:
  vector<value_t> H;
  loss_function->compute_H(H, theta, Ax, data);
  vector<value_t> hess_cache(working_set_size, 0.0);
  for (size_t ind = 0; ind < working_set_size; ++ind) {
    size_t j = prioritized_features[ind];
    hess_cache[ind] = data->get_column(j)->h_norm_sq(H);
  }

  // Cache values for updating intercept:
  value_t sum_theta = sum_vector(theta);
  value_t sum_H = sum_vector(H);

  // Finish set up for approximate newton step:
  value_t prox_newton_epsilon = 0.0;
  int max_cd_itr = MAX_PROX_NEWTON_CD_ITR;
  if (first_prox_newton_iteration) {
    max_cd_itr = MIN_PROX_NEWTON_CD_ITR;
    first_prox_newton_iteration = false;

    // Set up gradient values:
    prox_newton_grad_cache.resize(working_set_size);
    for (size_t ind = 0; ind < working_set_size; ++ind) {
      size_t j = prioritized_features[ind];
      prox_newton_grad_cache[ind] = data->get_column(j)->inner_product(theta);
    }
  } else {
    prox_newton_epsilon = PROX_NEWTON_EPSILON_RATIO * prox_newton_grad_diff;
  }
  
  // Approximately solve for newton direction:
  for (int cd_itr = 0; cd_itr < max_cd_itr; ++cd_itr) {

    // Shuffle indices:
    random_shuffle(rand_permutation.begin(), rand_permutation.end());
    for (size_t rp_ind = 0; rp_ind < working_set_size; ++rp_ind) {
      size_t new_index = rand_permutation[rp_ind];
      swap(prioritized_features[rp_ind], prioritized_features[new_index]);  
      swap(Delta_x[rp_ind], Delta_x[new_index]);
      swap(prox_newton_grad_cache[rp_ind], prox_newton_grad_cache[new_index]);
      swap(hess_cache[rp_ind], hess_cache[new_index]);
    }

    // Compute newton direction using coordinate descent:
    value_t sum_sq_hess_diff = 0.0;
    for (size_t ind = 0; ind < working_set_size; ++ind) {
      size_t j = prioritized_features[ind];
      const Column *col = data->get_column(j);

      // Compute hessian and gradient for coordinate j:
      value_t hess = hess_cache[ind];
      if (hess <= 0.0)
        continue;

      value_t grad = prox_newton_grad_cache[ind] 
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
      sum_sq_hess_diff += diff * diff * hess * hess;
    }

    // Update intercept:
    if (use_intercept) {
      value_t ip = inner_product(H, A_Delta_x);
      value_t diff = -(ip + sum_theta)/sum_H;
      Delta_intercept += diff;
      add_scaler(A_Delta_x, diff);
    }

    if (sum_sq_hess_diff < prox_newton_epsilon
        && cd_itr + 1 >= MIN_PROX_NEWTON_CD_ITR) {
      break;
    }
  }

  // Apply update with backtracking:
  value_t t = 1.0;
  value_t last_t = 0.0;
  for (int backtrack_itr = 0; backtrack_itr < MAX_BACKTRACK_ITR; ++backtrack_itr) {
    value_t diff_t = t - last_t;

    intercept += diff_t * Delta_intercept;
    value_t subgrad_t = 0.0;

    for (size_t ind = 0; ind < working_set_size; ++ind) {
      size_t j = prioritized_features[ind];
      x[j] += diff_t * Delta_x[ind];
      if (x[j] < 0)
        subgrad_t -= lambda * Delta_x[ind];
      else if (x[j] > 0)
        subgrad_t += lambda * Delta_x[ind];
      else
        subgrad_t -= lambda * fabs(Delta_x[ind]);
    }
    for (size_t i = 0; i < n; ++i) {
      Ax[i] += diff_t * A_Delta_x[i];
    }

    loss_function->compute_dual_points(Ax, theta, aux_dual, data);
    subgrad_t += inner_product(A_Delta_x, theta);

    if (subgrad_t < 0) {
      break;
    } else {
      last_t = t;
      t *= 0.5;
    }
  }

  // Update intercept exactly:
  value_t delta_intercept = update_intercept(intercept, loss_function, data);

  // Cache gradients for next iteration:
  if (t != 1.0) {
    for (size_t i = 0; i < A_Delta_x.size(); ++i) {
      A_Delta_x[i] = t * A_Delta_x[i] + delta_intercept;
    }
  }
  for (size_t ind = 0; ind < working_set_size; ++ind) {
    size_t j = prioritized_features[ind];
    const Column *col = data->get_column(j);
    value_t actual_grad = col->inner_product(theta);
    value_t approximate_grad = prox_newton_grad_cache[ind]
                             + col->weighted_inner_product(A_Delta_x, H);

    prox_newton_grad_cache[ind] = actual_grad;
    value_t diff = actual_grad - approximate_grad;
    prox_newton_grad_diff += diff * diff;
  }

  // Compute theta_scale return value:
  value_t max_grad_working_set = max_abs(prox_newton_grad_cache);
  if (max_grad_working_set > lambda)
    return lambda / max_grad_working_set;
  else
    return 1.0;

}

value_t Solver::priority_norm_j(size_t j, const Dataset* data) {
  // Returns l2 norm of feature j
  // If using intercept, returns l2 norm of A_j - mean(A_j)

  if (use_intercept)
    return data->get_column(j)->l2_norm_centered();
  else
    return data->get_column(j)->l2_norm();
}

value_t Solver::compute_alpha(const Dataset* data, value_t lambda, value_t theta_scale) {
  // Returns alpha, the largest value in [0,1] such that
  // (1 - alpha) * phi + alpha * theta * theta_scale is feasible
  // Requires values of ATphi and ATtheta to be current

  value_t best_alpha = 1.0;
  for (vector<size_t>::iterator j = prioritized_features.begin();
       j != prioritized_features.end();
       ++j) {

    value_t norm = priority_norm_j(*j, data);
    if (norm <= 0.0)
      continue;

    value_t l = ATphi[*j];
    value_t r = theta_scale * ATtheta[*j];
    if (fabs(r) <= lambda)
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
  // Updates values of ATtheta for all feature indices in
  // prioritized_features vector

  size_t d = data->get_d();
  ATtheta.resize(d);
  for (vector<size_t>::iterator j = prioritized_features.begin();
       j != prioritized_features.end();
       ++j) {
    const Column *col = data->get_column(*j);
    ATtheta[*j] = col->inner_product(theta);
  }
}

void Solver::update_phi(value_t alpha, value_t theta_scale) {
  // Updates phi via phi = (1-alpha)*phi + alpha*theta*theta_scale
  // Also updates ATphi
  // Requires values of ATtheta and ATphi to be current

  for (vector<size_t>::iterator j = prioritized_features.begin();
       j != prioritized_features.end();
       ++j) {
    ATphi[*j] = (1 - alpha) * ATphi[*j] + alpha * theta_scale * ATtheta[*j];
  }
  for (size_t i = 0; i < phi.size(); ++i)
    phi[i] = (1 - alpha) * phi[i] + alpha * theta_scale * theta[i];
}

void Solver::prioritize_features(const Dataset *data, value_t lambda, const value_t *x, size_t max_size_C) {
  // Reorders prioritized_features so that first max_size_C
  // elements are feature indices with highest priority in order

  size_t d = data->get_d();
  feature_priorities.resize(d);

  for (vector<size_t>::iterator j = prioritized_features.begin();
       j != prioritized_features.end();
       ++j) {
    if (x[*j] != 0) {
      feature_priorities[*j] = 0.0;
    } else {
      value_t norm = priority_norm_j(*j, data);
      if (norm <= 0) {
        feature_priorities[*j] = std::numeric_limits<value_t>::max();
      } else {
        value_t priority_value = (lambda - fabs(ATphi[*j])) / norm;
        feature_priorities[*j] = priority_value;
      }
    }
  }
  IndirectComparator cmp(feature_priorities);
  nth_element(
    prioritized_features.begin(),
    prioritized_features.begin() + max_size_C,
    prioritized_features.end(),
    cmp);
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
                   int &itr_counter,
                   const char *log_directory) {

  // Solves l1-regularized loss minimization problem

  Timer timer;
  Logger logger(log_directory);
  cout.precision(10);

  itr_counter = 0;
  Loss* loss_function = get_loss_function(loss_type);
  size_t n = data->get_n();
  size_t d = data->get_d();
  working_set_size = 0;
  prioritized_features.resize(d);
  for (size_t j = 0; j < d; ++j)
    prioritized_features[j] = j;

  // Initialize dual points:
  loss_function->compute_Ax(Ax, x, intercept, data);
  loss_function->compute_dual_points(Ax, theta, aux_dual, data);

  phi.assign(n, 0.0);
  ATphi.assign(d, 0.0);
  theta_scale = 1.0;

  // Update intercept (if using intercept term):
  update_intercept(intercept, loss_function, data);

  // Initialize objective values:
  value_t primal_loss = loss_function->primal_loss(theta, aux_dual, data);
  value_t l1_penalty = lambda * l1_norm(x, d);
  primal_obj = primal_loss + l1_penalty;

  // Main Blitz loop:
  while (++itr_counter) {
    value_t primal_obj_last = primal_obj;

    update_ATtheta(data);

    value_t alpha = compute_alpha(data, lambda, theta_scale);
    update_phi(alpha, theta_scale);

    value_t dual_obj = loss_function->dual_obj(phi, data);
    duality_gap = primal_obj - dual_obj;

    // Determine working set size:
    working_set_size = 2 * l0_norm(x, d);
    if (working_set_size < 100)
      working_set_size = 100;
    if (working_set_size > prioritized_features.size())
      working_set_size = prioritized_features.size();

    prioritize_features(data, lambda, x, working_set_size);

    // Eliminate features:
    value_t thresh = sqrt(2 * duality_gap / loss_function->L);
    IndirectExceedsThreshold exceeds_thresh(feature_priorities, thresh);
    prioritized_features.erase(
      remove_if(prioritized_features.begin(), 
                prioritized_features.end(),
                exceeds_thresh),
      prioritized_features.end()
    );
    if (working_set_size > prioritized_features.size())
      working_set_size = prioritized_features.size();
  
    // Solve subproblem:
    value_t epsilon = 0.3; 
    reset_prox_newton_variables();
    while (true) {
      value_t last_subproblem_obj = primal_obj;
      theta_scale = run_prox_newton_iteration(
                        x, intercept, lambda, loss_function, data); 

      primal_loss = loss_function->primal_loss(theta, aux_dual, data);
      l1_penalty = lambda * l1_norm(x, d);
      primal_obj = primal_loss + l1_penalty;
      value_t subprob_dual_obj = loss_function->dual_obj(theta, data, theta_scale);

      value_t subprob_duality_gap = primal_obj - subprob_dual_obj;
      if (subprob_duality_gap < epsilon * (primal_obj - dual_obj))
        break;
      if (subprob_duality_gap / fabs(subprob_dual_obj) < tolerance)
        break;
      if (primal_obj >= last_subproblem_obj)
        break;
    }

    primal_loss = loss_function->primal_loss(theta, aux_dual, data);
    l1_penalty = lambda * l1_norm(x, d);
    primal_obj = primal_loss + l1_penalty;
    duality_gap = primal_obj - dual_obj;

    // Print/record results:
    double elapsed_time = timer.elapsed_time();
    timer.pause_timing();
    if (verbose)
      cout << "Time: " << elapsed_time
           << " Objective: " << primal_obj 
           << " Duality gap: " << duality_gap 
           << " Features left: " << prioritized_features.size() << endl;

    logger.log_point(elapsed_time, primal_obj);
    timer.continue_timing();

    // Test stopping conditions:
    if ((duality_gap / fabs(dual_obj) < tolerance) &&
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

}
