#pragma once 

#include "common.h"
#include "loss.h"
#include <vector>

namespace BlitzL1 {

  class Solver {
    private:
      std::vector<value_t> theta;
      std::vector<value_t> Ax;
      std::vector<value_t> ATtheta;
      std::vector<value_t> phi;
      std::vector<value_t> ATphi;
      std::vector<value_t> aux_dual;
      value_t theta_scale;

      size_t working_set_size;
      std::vector<size_t> prioritized_features;
      std::vector<value_t> feature_priorities;

      value_t tolerance;
      bool verbose;
      bool use_intercept;
      value_t min_time;
      value_t max_time;

      value_t update_intercept(value_t &intercept, 
                            const Loss *loss_function,
                            const Dataset *data);

      void prioritize_features(const Dataset *data, value_t lambda, const value_t *x, size_t max_size_C);

      bool first_prox_newton_iteration;
      value_t prox_newton_grad_diff;
      std::vector<value_t> prox_newton_grad_cache;
      void reset_prox_newton_variables();
      value_t run_prox_newton_iteration(value_t *x, 
                                     value_t &intercept, 
                                     value_t lambda,
                                     const Loss *loss_function, 
                                     const Dataset *data);

      void update_ATtheta(const Dataset *data);
      void update_phi(value_t alpha, value_t theta_scale);
      value_t compute_alpha(const Dataset* data, value_t lambda, value_t theta_scale);
      value_t priority_norm_j(size_t j, const Dataset* data);

    public:
      Solver() {
        tolerance = 0.0001;
        use_intercept = true;
        verbose = false;
        min_time = -60.0;
        max_time = 3.15569e9;
      }

      void set_tolerance(value_t val) { tolerance = val; }
      value_t get_tolerance() { return tolerance; }
      void set_max_time(value_t val) { max_time = val; }
      value_t get_max_time() { return max_time; }
      void set_min_time(value_t val) { min_time = val; }
      value_t get_min_time() { return min_time; }
      void set_use_intercept(bool val) { use_intercept = val; }
      bool get_use_intercept() { return use_intercept; }
      void set_verbose(bool val) { verbose = val; }
      bool get_verbose() { return verbose; }

      void solve(const Dataset *data,
                 value_t lambda,
                 const char *loss_type,
                 value_t *x,
                 value_t &intercept,
                 char *solution_status,
                 value_t &primal_obj,
                 value_t &duality_gap,
                 int &itr_counter,
                 const char* log_directory);

      value_t compute_lambda_max(const Dataset *data, const char* loss_type);
  };

}
