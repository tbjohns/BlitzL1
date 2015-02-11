#!/usr/bin/env python 

import os
import sys
import numpy as np
from scipy import sparse

pwd = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(pwd, "../.."))

import python as blitzl1

def approx_equal(x, y):
  if abs(x - y) < 1e-5:
    return True
  return False

def test_SmallLogReg():
  blitzl1.set_use_intercept(False)
  blitzl1.set_tolerance(0.0)
  blitzl1.set_verbose(False)
  A = np.arange(20).reshape(5, 4)
  b = np.arange(5)
  A = sparse.csc_matrix(A)
  prob = blitzl1.LogRegProblem(A, b)
  sol = prob.solve(2)
  if not approx_equal(sol.obj, 1.3220246925865164):
    print "test SmallLogReg obj failed"

def main():
  test_SmallLogReg()

main()
