import python as blitzl1

import numpy as np
from scipy import sparse

n = 10
d = 3
A = np.random.randn(n, d)
b = np.random.randn(n)
A_csc = sparse.csc_matrix(A)

prob = blitzl1.LassoProblem(A_csc, b)

prob.solve(0.1)

