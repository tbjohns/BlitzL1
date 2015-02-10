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

  c->add_multiple(vec, -2.0);
  if (vec[9] != 0.0 || vec[0] != -3.0)
    cerr << "Test ColumnFromPointers add_multiple failed" << endl;
}

void test_DatasetFromCSCPointers() {
  nnz_t indptr[6] = {0, 1, 2, 4, 6, 7};
  index_t indices[7] = {1, 0, 1, 2, 1, 2, 0};
  value_t values[7] = {-2, 1, 3, 1, 1, 2, 1};
  value_t labels[3] = {-1, 0.5, 1.5};
  Dataset *data = new DatasetFromCSCPointers(indices, indptr, values, labels, 3, 5, 7);

  vector<value_t> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  if (data->get_column(0)->l2_norm_sq() != 4.0)
    cerr << "Test DatasetFromCSCPointers column 0 failed" << endl;
  if (data->get_column(1)->inner_product(vec) != 1.0)
    cerr << "Test DatasetFromCSCPointers column 1 failed" << endl;
  if (data->get_column(2)->inner_product(vec) != 9.0)
    cerr << "Test DatasetFromCSCPointers column 2 failed" << endl;
  if (data->get_column(3)->mean() != 1.0)
    cerr << "Test DatasetFromCSCPointers column 3 failed" << endl;
  if (data->get_column(4)->inner_product(vec) != 1.0)
    cerr << "Test DatasetFromCSCPointers column 4 failed" << endl;
  if (data->get_label(0) != -1.0)
    cerr << "Test DatasetFromCSCPointers label 0 failed" << endl;
  if (data->get_label(2) != 1.5)
    cerr << "Test DatasetFromCSCPointers last label failed" << endl;
}

void test_Math() {
  vector<value_t> vec;
  vec.assign(10, 0.0);
  vec[0] = -5.0;
  vec[3] = 2.0;
  vec[9] = 12.0;
  if (l2_norm_sq(vec) != 173.0)
    cerr << "Test Math l2_norm_sq failed" << endl;

  vector<value_t> vec2;
  vec2.assign(10, 1.0);
  vec2[9] = -1.0;
  if (inner_product(vec2, vec) != -15.0)
    cerr << "Test Math inner_product failed" << endl;

  vector<value_t> vec3;
  vec3.push_back(-1.0);
  vec3.push_back(-7.8);
  vec3.push_back(0.0);
  vec3.push_back(4.4);
  if (max_abs(vec3) != 7.8)
    cerr << "Test Math max_abs failed" << endl;

  if (soft_threshold(-5.5, 5.5) != 0.0 ||
      soft_threshold(-5.5, 2.0) != -3.5 ||
      soft_threshold(-3, 5.5) != 0.0 ||
      soft_threshold(6, 2) != 4.0)
    cerr << "Test Math soft_threshold failed" << endl;

  value_t x[5] = {-1.0, -1.5, 0.0, 0.0, 3.3};
  if (l1_norm(x, 5) != 5.8)
    cerr << "Test Math l1_norm failed" << endl; 

}

int main() {
  test_ColumnFromPointers();
  test_DatasetFromCSCPointers();
  test_Math();
  return 0;
}
