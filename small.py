#!/usr/bin/env python 


import python as blitzl1

import numpy as np
from scipy import sparse

n = 100
d = 100
A = np.random.randn(n, d)
#b = np.random.randn(n)
b = 2.0 * (np.random.randn(n) < 0) - 1.0
print b
A_csc = sparse.csc_matrix(A)

blitzl1.set_tolerance(0.2)
print "tolerance is", blitzl1.get_tolerance()

#prob = blitzl1.LassoProblem(A_csc, b)
prob = blitzl1.LogRegProblem(A_csc, b)

sol = prob.solve(0.1)

from IPython import embed
embed()

