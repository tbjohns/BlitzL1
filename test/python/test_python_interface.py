#!/usr/bin/env python 

import os
import sys
import numpy as np
from scipy import sparse

pwd = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(pwd, "../.."))

import python as blitzl1

def test_DataLoad():

  n = 10
  d = 200
  A = np.arange(n*d, dtype=np.float).reshape(n, d)
  b = np.arange(n, dtype=np.float)

  col_norm_0 = np.linalg.norm(A[:,0])
  col_norm_last = np.linalg.norm(A[:,d-1])

  B = np.arange(n*d, dtype=np.float).reshape(n, d)
  prob2 = blitzl1.LassoProblem(B, b)
  #if prob._get_A_column_norm(0) != col_norm_0:
  #  print "Dense data load failed (col_norm_0)"
  #if prob._get_A_column_norm(d-1) != col_norm_last:
  #  print "Dense data load failed (col_norm_last)"
  #if prob._get_label_i(n-1) != b[n-1]:
  #  print "Dense labels load failed"

  #A_float16 = np.array(A, dtype=np.float16)
  #b_float16 = np.array(b, dtype=np.float16)
  #prob = blitzl1.LassoProblem(A_float16, b_float16)
  #if prob._get_A_column_norm(0) != col_norm_0:
  #  print "Dense float16 data load failed (col_norm_0)"
  #if prob._get_A_column_norm(d-1) != col_norm_last:
  #  print "Dense float16 data load failed (col_norm_last)"
  #if prob._get_label_i(n-1) != b[n-1]:
  #  print "Dense float16 labels load failed"

  A_csc = sparse.csc_matrix(A)
  prob = blitzl1.LassoProblem(A_csc, b)
  if prob._get_A_column_norm(0) != col_norm_0:
    print "CSC data load failed (col_norm_0)"
  if prob._get_A_column_norm(d-1) != col_norm_last:
    print "CSC data load failed (col_norm_last)"
  if prob._get_label_i(n-1) != b[n-1]:
    print "CSC labels load failed"

  A_csr = sparse.csr_matrix(A)
  prob = blitzl1.LassoProblem(A_csr, b)
  if prob._get_A_column_norm(0) != col_norm_0:
    print "CSR data load failed (col_norm_0)"
  if prob._get_A_column_norm(d-1) != col_norm_last:
    print "CSR data load failed (col_norm_last)"
  if prob._get_label_i(n-1) != b[n-1]:
    print "CSR labels load failed"

  A_float16 = sparse.csr_matrix(A, dtype=np.float16)
  prob = blitzl1.LassoProblem(A_float16, b)
  diff = abs(prob._get_A_column_norm(d-1) - col_norm_last)
  if diff > 1.0:
    print "CSR float16 data load failed (col_norm_last)"

def test_SolverOptions():
  blitzl1.set_tolerance(0.027)  
  if blitzl1.get_tolerance() != 0.027:
    print "test SolverOptions tolerance failed"

  blitzl1.set_max_time(557.0)
  if blitzl1.get_max_time() != 557.0:
    print "test SolverOptions max_time failed"

  blitzl1.set_use_intercept(True)
  if blitzl1.get_use_intercept() != True:
    print "test SolverOptions use_intercept (True) failed"

  blitzl1.set_use_intercept(False)
  if blitzl1.get_use_intercept() != False:
    print "test SolverOptions use_intercept (False) failed"

  blitzl1.set_verbose(True)
  if blitzl1.get_verbose() != True:
    print "test SolverOptions verbose (True) failed"

  blitzl1.set_verbose(False)
  if blitzl1.get_verbose() != False:
    print "test SolverOptions verbose (False) failed"

def main():
  test_DataLoad()
  test_SolverOptions()
  return 0


main()
