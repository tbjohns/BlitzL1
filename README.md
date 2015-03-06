# BlitzL1

BlitzL1 is a fast, scalable library for minimizing L1-regularized losses.  L1-regularized learning is widely used in statistics and machine learning as it fits a function to data while simultaneously encouraging the result to be sparse (only a fraction of features used for prediction).  

Specifically, Blitz solves the following problems:

Name                       | Objective
-------------------------- | ------------------------------
Lasso                      | ![Lasso objective](images/lasso.png)
Sparse logistic regression | ![Logistic regression objective](images/logreg.png)

On a single machine, Blitz can be called from Python or C++ with support for additional languages coming soon.  Calls to Blitz have low overhead (minimal memory copying), meaning Blitz can be used as an effective subproblem solver in more elaborate algorithms.

For larger problems, we are also working on releasing out-of-core and distributed implementations.

## Use with Python

You can try out an early version of the code using Python.  To install, run `pip install blitzl1`.  The following solves a sparse logistic regression problem with regularization λ=1:
```
import blitzl1
prob = blitzl1.LogRegProblem(A, b)
sol = prob.solve(1.0)
```
Of course there is much more than that, so please play around.  Early feedback is much appreciated!
