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

def test_SimpleLasso():
  blitzl1.set_use_intercept(False)
  blitzl1.set_tolerance(0.0)
  blitzl1.set_verbose(False)
  A = np.eye(4)
  A[3,3] = 2.0
  A[2,2] = 2.0
  b = np.array([5., -2., 2., -6.])
  A = sparse.csc_matrix(A)
  prob = blitzl1.LassoProblem(A, b)
  sol = prob.solve(1)
  if not approx_equal(sol.x[0], 4.0) or not approx_equal(sol.x[3], -2.75):
    print "test SimpleLasso failed"

def test_SmallLasso():
  blitzl1.set_use_intercept(False)
  blitzl1.set_tolerance(0.0)
  blitzl1.set_verbose(False)
  A = np.arange(20).reshape(5, 4)
  b = np.arange(5)
  A = sparse.csc_matrix(A)
  prob = blitzl1.LassoProblem(A, b)
  sol = prob.solve(2)
  if not approx_equal(sol.obj, 0.4875):
    print "test SmallLasso obj failed"

def main():
  test_SimpleLasso()
  test_SmallLasso()


main()
