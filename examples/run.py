import blitzl1

import sys
import os
from scipy import sparse
import numpy as np
from sklearn.datasets import load_svmlight_file

blitzl1.set_verbose(True)

def format_b(b):
  max_b = max(b)
  min_b = min(b)
  scale = 2.00 / (max_b - min_b)
  return scale * (b - max_b) + 1.0

(A, b) = load_svmlight_file(os.path.join(pwd, "../benchmark/data/news20"))
A_csc = sparse.csc_matrix(A)
b = format_b(b)

from IPython import embed
embed()

prob = blitzl1.LogRegProblem(A_csc, b)

lammax = prob.compute_lambda_max()
sol = prob.solve(0.001 * lammax)

from IPython import embed
embed()
