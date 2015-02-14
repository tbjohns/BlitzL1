#!/usr/bin/env python -W ignore

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
  b = np.array([1, -1, -1, 1, 1])
  A = sparse.csc_matrix(A)
  prob = blitzl1.LogRegProblem(A, b)
  sol = prob.solve(2)
  if not approx_equal(sol.objective_value, 3.312655451335882):
    print "test SmallLogReg obj failed"
  if not approx_equal(sol.x[0], 0.0520996109147):
    print "test SmallLogReg x[0] failed"

  python_obj = sol.evaluate_loss(A, b) + 2 * np.linalg.norm(sol.x, ord=1)
  if not approx_equal(sol.objective_value, python_obj):
    print "test SmallLogReg python_obj failed"

  blitzl1.set_use_intercept(True)
  blitzl1.set_tolerance(0.0001)
  sol = prob.solve(1.5)

  blitzl1.set_tolerance(0.01)
  sol2 = prob.solve(1.5, initial_x=sol.x, initial_intercept=sol.intercept)
  if sol2._num_iterations != 1:
    print "test SmallLogReg initial conditions failed"

def main():
  test_SmallLogReg()

main()
