#!/usr/bin/env python 

import os
import sys
import numpy as np
from scipy import sparse

pwd = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(pwd, "../../python"))

import l1_problem

def test_DataLoad():

  n = 10
  d = 12
  A = np.random.randn(n, d)
  b = np.random.randn(n)

  col_norm_0 = np.linalg.norm(A[:,0])
  col_norm_last = np.linalg.norm(A[:,d-1])

  A_csc = sparse.csc_matrix(A)
  prob = l1_problem.L1Problem(A_csc, b)
  if prob._get_A_column_norm(0) != col_norm_0:
    print "CSC data load failed (col_norm_0)"
  if prob._get_A_column_norm(d-1) != col_norm_last:
    print "CSC data load failed (col_norm_last)"
  if prob._get_label_i(n-1) != b[n-1]:
    print "CSC labels load failed"

  A_csr = sparse.csr_matrix(A)
  prob = l1_problem.L1Problem(A_csr, b)
  if prob._get_A_column_norm(0) != col_norm_0:
    print prob._get_A_column_norm(0)
    print col_norm_0
    print "CSR data load failed (col_norm_0)"
  if prob._get_A_column_norm(d-1) != col_norm_last:
    print "CSR data load failed (col_norm_last)"
  if prob._get_label_i(n-1) != b[n-1]:
    print "CSR labels load failed"

  A_float16 = sparse.csr_matrix(A, dtype=np.float16)
  prob = l1_problem.L1Problem(A_float16, b)
  diff = abs(prob._get_A_column_norm(d-1) - col_norm_last)
  if diff > 0.1:
    print "float16 data load failed (col_norm_last)"

def main():
  test_DataLoad()


main()
