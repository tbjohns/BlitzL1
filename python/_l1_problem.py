import os
import numpy as np 
import ctypes
from scipy import sparse

_dir = os.path.abspath(os.path.dirname(__file__))
_lib = np.ctypeslib.load_library("../lib/libblitzl1.so", _dir)


_index_t = ctypes.c_int32
_value_t = ctypes.c_double
_size_t = ctypes.c_int32
_pointer = ctypes.POINTER(ctypes.c_void_p)
_value_t_p = ctypes.POINTER(_value_t)
_index_t_p = ctypes.POINTER(_index_t)
_size_t_p = ctypes.POINTER(_size_t)
_char_p = ctypes.c_char_p
_bool = ctypes.c_bool


_lib.BlitzL1_new_sparse_dataset.restype = _pointer
_lib.BlitzL1_new_sparse_dataset.argtypes = [_index_t_p, _size_t_p, _value_t_p, _value_t_p, _index_t, _index_t, _size_t]
_lib.BlitzL1_get_column_norm.restype = _value_t
_lib.BlitzL1_get_column_norm.argtype = [_pointer, _index_t]
_lib.BlitzL1_get_label_i.restype = _value_t
_lib.BlitzL1_get_label_i.argtype = [_pointer, _index_t]
_lib.BlitzL1_new_solver.restype = _pointer
_lib.BlitzL1_new_solver.argtype = None
_lib.BlitzL1_solve_problem.restype = None
_lib.BlitzL1_solve_problem.argtype = [_pointer, _pointer, _value_t, _char_p, _value_t_p, _value_t, _value_t, _value_t, _char_p]
_lib.BlitzL1_set_tolerance.restype = None
_lib.BlitzL1_set_tolerance.argtype = [_pointer, _value_t]
_lib.BlitzL1_get_tolerance.restype = _value_t
_lib.BlitzL1_get_tolerance.argtype = None
_lib.BlitzL1_set_max_time.restype = None
_lib.BlitzL1_set_max_time.argtype = [_pointer, _value_t]
_lib.BlitzL1_get_max_time.restype = _value_t
_lib.BlitzL1_get_max_time.argtype = None
_lib.BlitzL1_set_use_intercept.restype = None
_lib.BlitzL1_set_use_intercept.argtype = [_pointer, _bool]
_lib.BlitzL1_get_use_intercept.restype = _bool
_lib.BlitzL1_get_use_intercept.argtype = None
_lib.BlitzL1_set_verbose.restype = None
_lib.BlitzL1_set_verbose.argtype = [_pointer, _bool]
_lib.BlitzL1_get_verbose.restype = _bool
_lib.BlitzL1_get_verbose.argtype = None

_solver = _lib.BlitzL1_new_solver()

def set_tolerance(value):
  _lib.BlitzL1_set_tolerance(_solver, _value_t(value))

def get_tolerance():
  return _lib.BlitzL1_get_tolerance(_solver)

def set_max_time(value):
  _lib.BlitzL1_set_max_time(_solver, _value_t(value))

def get_max_time():
  return _lib.BlitzL1_get_max_time(_solver)

def set_use_intercept(value):
  _lib.BlitzL1_set_use_intercept(_solver, _bool(value))

def get_use_intercept():
  return _lib.BlitzL1_get_use_intercept(_solver)

def set_verbose(value):
  _lib.BlitzL1_set_verbose(_solver, _bool(value))

def get_verbose():
  return _lib.BlitzL1_get_verbose(_solver)

def data_as(obj, ctypes_type, force_return_obj=False):
  return_obj = obj
  if obj.dtype != ctypes_type:
    obj = obj.astype(ctypes_type)
    return_obj = obj
  return (return_obj, obj.ctypes.data_as(ctypes_type))



class _L1Problem(object):
  def __init__(self, A, b):
    self._load_dataset(A, b)

  def _load_dataset(self, A, b):
    self.shape = A.shape
    if sparse.issparse(A):
      if sparse.isspmatrix_csc(A):
        format_changed = False
      else:
        A = A.tocsc()
        format_changed = True

      (self.indices, indices_arg) = data_as(A.indices, _index_t_p, format_changed)
      (self.indptr, indptr_arg) = data_as(A.indptr, _size_t_p, format_changed)
      (self.data, data_arg) = data_as(A.data, _value_t_p, format_changed)
      (self.b, labels_arg) = data_as(b, _value_t_p)
      n = _index_t(A.shape[0])
      d = _index_t(A.shape[1])
      nnz = _size_t(A.nnz)
      self.dataset = _lib.BlitzL1_new_sparse_dataset(
          indices_arg, indptr_arg, data_arg, labels_arg, n, d, nnz)

  def _get_A_column_norm(self, j):
    return _lib.BlitzL1_get_column_norm(self.dataset, _index_t(j))

  def _get_label_i(self, i):
    return _lib.BlitzL1_get_label_i(self.dataset, _index_t(i))

  def solve(self, lam, log_dir=""):
    lambda_arg = _value_t(lam)
    (n, d) = self.shape
    (x, x_arg) = data_as(np.zeros(d), _value_t_p)
    intercept_arg = _value_t()
    obj_arg = _value_t()
    duality_gap_arg = _value_t()
    loss_arg = _char_p(self.LOSS_TYPE)
    log_dir_arg = _char_p(log_dir)
    _lib.BlitzL1_solve_problem(_solver, self.dataset, lambda_arg, loss_arg, x_arg, ctypes.byref(intercept_arg), ctypes.byref(obj_arg), ctypes.byref(duality_gap_arg), log_dir_arg)
    return self.SOLUTION_TYPE(x, intercept_arg.value, obj_arg.value, duality_gap_arg.value)
    
class _Solution(object):
  def __init__(self, x, intercept, obj, duality_gap):
    self.x = x
    self.intercept = intercept
    self.obj = obj
    self.duality_gap = duality_gap

  def _compute_Ax(self, A):
    if sparse.issparse(A):
      result = A * np.mat(self.x).T + self.intercept
      return np.array(result).flatten()
    else:
      return np.dot(A, self.x) + self.intercept

class LassoSolution(_Solution):
  def predict(self, A):
    return self._compute_Ax(A)

  def evaluate_loss(self, A, b):
    predictions = self.predict(A)
    return 0.5 * np.linalg.norm(b - predictions) ** 2

class LogRegSolution(_Solution):
  def predict(self, A):
    Ax = self._compute_Ax(A)
    return 1 / (1 + np.exp(-Ax))

  def evaluate_loss(self, A, b):
    exp_mbAx = np.exp(-b * self._compute_Ax(A))
    return sum(np.log1p(exp_mbAx))

class LassoProblem(_L1Problem):
  LOSS_TYPE = "squared"
  SOLUTION_TYPE = LassoSolution

class LogRegProblem(_L1Problem):
  LOSS_TYPE = "logistic"
  SOLUTION_TYPE = LogRegSolution


