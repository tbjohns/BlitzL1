#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy import sparse
import python as blitzl1

#def bug():
n = 10
d = 200

B = np.arange(n*d, dtype=np.float).reshape(n, d)
b = np.arange(n, dtype=np.float)

print B[0,0]
prob = blitzl1.LassoProblem(B, b)

#bug()
