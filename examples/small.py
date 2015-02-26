import sys
import os
pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, ".."))

import python as blitzl1
import numpy as np
from scipy import sparse

blitzl1.set_verbose(True)

n = 2000
d = 20000

A = np.random.randn(n, d)
A = sparse.csc_matrix(A)
b = np.random.randn(n)

from IPython import embed
embed()

prob = blitzl1.LassoProblem(A, b)
lammax = prob.compute_lambda_max()
sol = prob.solve(lammax * 0.0001)

from IPython import embed
embed()
