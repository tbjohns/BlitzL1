import blitzl1

import sys
import os
import numpy as np
from scipy import sparse

blitzl1.set_verbose(True)
blitzl1.set_tolerance(0.0)

n = 100
d = 1000

A = np.random.randn(n, d)
A = sparse.csc_matrix(A)
b = np.random.randn(n)

prob = blitzl1.LassoProblem(A, b)
lammax = prob.compute_lambda_max()
print "lammax is", lammax
sol = prob.solve(lammax * 0.1)

from IPython import embed
embed()
