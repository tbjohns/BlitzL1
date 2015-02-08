#include "../../src/common.h"

#include <math.h>
#include <vector>
#include <iostream>
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

using namespace BlitzL1;

void test_column() {
  value_t values[3] = {2.0, 3.0, 5.0};
  index_t indices[3] = {0, 1, 9};
  index_t nnz = 3;
  index_t length = 10;
  Column *c = new ColumnFromPointers(indices, values, nnz, length);

  vector<value_t> vec;
  for (index_t i = 1; i <= length; ++i)
    vec.push_back((value_t) i);

  if (c->mean() != 1.0)
    cerr << "Test ColumnFromPointers mean failed" << endl;

  if (c->l2_norm_sq() != 38.0)
    cerr << "Test ColumnFromPointers l2_norm_sq failed" << endl;

  if (c->l2_norm() != sqrt(38.0))
    cerr << "Test ColumnFromPointers l2_norm failed" << endl;

  if (c->l2_norm_centered() != sqrt(28.0)) 
    cerr << "Test ColumnFromPointers l2_norm_centered failed" << endl;

  if (c->inner_product(vec) != 58.0)
    cerr << "Test ColumnFromPointers inner_product failed" << endl;

  if (c->h_norm_sq(vec) != 272.0)
    cerr << "Test ColumnFromPointers h_norm_sq failed" << endl;
}

int main() {
  test_column();
  return 0;
}
