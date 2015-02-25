import python as blitzl1
import numpy as np

blitzl1.set_verbose(True)

n = 200
d = 2000

A = np.random.randn(n, d)
b = np.random.randn(n)

prob = blitzl1.LassoProblem(A, b)
lammax = prob.compute_lambda_max()
sol = prob.solve(lammax * 0.05)
