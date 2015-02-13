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
    print "test SimpleLasso basic failed"

  blitzl1.set_use_intercept(True)
  sol = prob.solve(1)
  if not approx_equal(sol.intercept, -0.25):
    print "test SimpleLasso intercept failed"

  if not approx_equal(sol.objective_value, 9.75):
    print "test SimpleLasso obj failed"

  python_obj = sol.evaluate_loss(A, b) + np.linalg.norm(sol.x, ord=1)
  if not approx_equal(sol.objective_value, python_obj):
    print "test SimpleLasso python_obj failed"


def test_SmallLasso():
  blitzl1.set_use_intercept(False)
  blitzl1.set_tolerance(0.0)
  blitzl1.set_verbose(False)
  A = np.arange(20).reshape(5, 4)
  b = np.arange(5)
  A = sparse.csc_matrix(A)
  prob = blitzl1.LassoProblem(A, b)
  sol = prob.solve(2)
  if not approx_equal(sol.objective_value, 0.4875):
    print "test SmallLasso obj failed"

  save_path = "/tmp/blitzl1_save_test"
  sol.save(save_path)
  sol2 = blitzl1.load_solution(save_path)
  if not np.all(sol.x == sol2.x):
    print "test SmallLasso save_x failed"
  if sol.objective_value != sol2.objective_value:
    print "test SmallLasso save_obj failed"
  os.remove(save_path)


  blitzl1.set_tolerance(0.1)
  log_path = "/tmp/blitzl1_log_test/"
  sol = prob.solve(5.0, log_directory=log_path)
  log_point = 0
  while True:
    time_file = "%s/time.%d" % (log_path, log_point)
    obj_file = "%s/obj.%d" % (log_path, log_point)
    try:
      time = float(open(time_file).read())
      obj = float(open(obj_file).read())
    except:
      break
    log_point += 1
  if obj != sol.objective_value:
    print "test SmallLasso log_obj failed"
  if time <= 0.0:
    print "test SmallLasso log_time failed"
  os.system("rm -r %s" % log_path)

def main():
  test_SimpleLasso()
  test_SmallLasso()


main()
