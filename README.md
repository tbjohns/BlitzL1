# BlitzL1

BlitzL1 is an efficient library for L1-regularized loss minimization in early stages of development.  Blitz can be called from Python with minimal overhead (support for R to come later).  For larger problems, we also intend to provide a distributed implementation using [rabit](https://github.com/tqchen/rabit).

## Try it out

Currently the easiest way to try Blitz is through its Python wrapper.  First download the repository and run `make`.  The following solves a sparse logistic regression problem with regularization 1.0:
```
import python as blitzl1
prob = blitzl1.LogRegProblem(A, b)
sol = prob.solve(1.0)
```
Of course, there are more features than that, so please play around.  Early feedback is much appreciated!
