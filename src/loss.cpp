#include "loss.h"

using namespace BlitzL1;

void Loss::compute_Ax(
        const value_t *x,
        value_t intercept,
        const Dataset *data,
        std::vector<value_t> &result) const {

  index_t n = data->get_n();
  index_t d = data->get_d();
  result.assign(n, intercept);
  for (index_t j = 0; j < d; ++j) {
    if (x[j] != 0)
      data->get_column(j)->add_multiple(result, x[j]);
  }
}
