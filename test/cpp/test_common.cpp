#include "../../src/common.h"

#include <math.h>
#include <vector>
#include <iostream>
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

using namespace BlitzL1;

void test_ColumnFromPointers() {
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

void test_CSCDatasetFromPointers() {
  size_t indptr[6] = {0, 1, 2, 4, 6, 7};
  index_t indices[7] = {1, 0, 1, 2, 1, 2, 0};
  value_t values[7] = {-2, 1, 3, 1, 1, 2, 1};
  value_t labels[3] = {-1, 0.5, 1.5};
  Dataset *data = new CSCDatasetFromPointers(indices, indptr, values, labels, 3, 5, 7);

  vector<value_t> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  if (data->get_column(0)->l2_norm_sq() != 4.0)
    cerr << "Test CSCDatasetFromPointers column 0 failed" << endl;
  if (data->get_column(1)->inner_product(vec) != 1.0)
    cerr << "Test CSCDatasetFromPointers column 1 failed" << endl;
  if (data->get_column(2)->inner_product(vec) != 9.0)
    cerr << "Test CSCDatasetFromPointers column 2 failed" << endl;
  if (data->get_column(3)->mean() != 1.0)
    cerr << "Test CSCDatasetFromPointers column 3 failed" << endl;
  if (data->get_column(4)->inner_product(vec) != 1.0)
    cerr << "Test CSCDatasetFromPointers column 4 failed" << endl;
  if (data->get_label(0) != -1.0)
    cerr << "Test CSCDatasetFromPointers label 0 failed" << endl;
  if (data->get_label(2) != 1.5)
    cerr << "Test CSCDatasetFromPointers last label failed" << endl;
}

int main() {
  test_ColumnFromPointers();
  test_CSCDatasetFromPointers();
  cout << "Common tests completed" << endl;
  return 0;
}
