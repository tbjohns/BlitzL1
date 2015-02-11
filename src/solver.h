#pragma once 

#include "common.h"
#include <vector>

namespace BlitzL1 {

  class Solver {
    private:
      std::vector<value_t> theta;
      std::vector<value_t> ATtheta;
      std::vector<value_t> phi;
      std::vector<value_t> ATphi;
      std::vector<value_t> aux_dual;

      std::vector<index_t> prioritized_features;
      std::vector<index_t> feature_priorities;

      value_t tolerance;
      bool verbose;
      bool use_intercept;
      value_t min_time;
      value_t max_time;

    public:
      Solver() {}

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

      void solve(Dataset *data,
                 value_t lambda,
                 char* loss_type,
                 value_t* x,
                 value_t &intercept,
                 value_t &primal_obj,
                 value_t &duality_gap,
                 char* log_directory);
  };

}
