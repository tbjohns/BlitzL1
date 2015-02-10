#include "../../src/common.h"
#include "../../src/loss.h"

#include <vector>
#include <iostream>
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

using namespace BlitzL1;

void test_SquaredLoss() {

  Loss *loss = new SquaredLoss();

  nnz_t indptr[6] = {0, 1, 2, 4, 6, 7};
  index_t indices[7] = {1, 0, 1, 2, 1, 2, 0};
  value_t values[7] = {-2, 1, 3, 1, 1, 2, 1};
  value_t labels[3] = {-1, 0.5, 1.5};
  Dataset *data = new DatasetFromCSCPointers(indices, indptr, values, labels, 3, 5, 7);

  vector<value_t> theta;
  vector<value_t> aux_dual;
  value_t x[5] = {1.0, 0.0, 0.0, 0.0, -2.0};
  value_t intercept = 2.0;
  loss->compute_dual_points(theta, aux_dual, x, intercept, data);
  if (theta[0] != 1.0 || theta[1] != -0.5 || theta[2] != 0.5)
    cerr << "Test SquaredLoss compute_dual_points failed" << endl;
  
  if (loss->primal_loss(theta, aux_dual) != 0.75)
    cerr << "Test SquaredLoss primal_loss failed" << endl;

  if (loss->dual_obj(theta, data) != -0.25)
    cerr << "Test SquaredLoss dual_obj failed" << endl;

  if (loss->L != 1.0)
    cerr << "Test SquaredLoss L failed" << endl;
}

int main() {
  test_SquaredLoss();
  return 0;
}
