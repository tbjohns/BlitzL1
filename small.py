import python as blitzl1
import numpy as np
A = np.random.randn(1000,100)
b = np.random.randn(1000)

prob = blitzl1.LassoProblem(A, b)
prob.solve(1.0)
from IPython import embed
embed()
