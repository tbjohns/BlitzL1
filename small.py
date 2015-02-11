#!/usr/bin/env python 


import python as blitzl1

import numpy as np
from scipy import sparse

#n = 100
#d = 100
#A = np.random.randn(n, d)
#b = np.random.randn(n)
#b = 2.0 * (np.random.randn(n) < 0) - 1.0
A = np.arange(20).reshape(5, 4)

b = np.arange(5)


A_csc = sparse.csc_matrix(A)

blitzl1.set_tolerance(0.0)

prob = blitzl1.LogRegProblem(A_csc, b)

sol = prob.solve(2)

print sol.evaluate_loss(A, b) + 2 * np.linalg.norm(sol.x, ord=1)

from IPython import embed

embed()


