#include "common.h"
#include "problem.h"

#include <vector>

namespace BlitzL1 {

  class Solver {
    protected:
      bool verbose;
      value_t min_time;
      value_t max_time;
      
      std::vector<value_t> theta;
      std::vector<value_t> ATtheta;
      std::vector<value_t> phi;
      std::vector<value_t> ATphi;

      std::vector<bool> is_eliminated;
      std::vector<index_t> prioritized_features;

    public:
      void solve(const Problem &prb,
                 value_t lambda,
                 value_t &x,
                 value_t &intercept,
                 char* log_directory);
      
  };

}
