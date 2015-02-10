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

      bool verbose;
      value_t min_time;
      value_t max_time;

    public:
      Solver() {}

      void solve(Dataset *data,
                 value_t lambda,
                 char* loss_type,
                 value_t* x,
                 value_t &intercept,
                 char* log_directory);
  };

}
