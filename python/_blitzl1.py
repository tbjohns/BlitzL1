import os
import numpy as np 
import ctypes
from scipy import sparse
import pickle

_dir = os.path.abspath(os.path.dirname(__file__))
_lib = np.ctypeslib.load_library("libblitzl1", _dir)

_index_t = ctypes.c_int32
_value_t = ctypes.c_double
_size_t = ctypes.c_int32
_pointer = ctypes.POINTER(ctypes.c_void_p)
_value_t_p = ctypes.POINTER(_value_t)
_index_t_p = ctypes.POINTER(_index_t)
_size_t_p = ctypes.POINTER(_size_t)
_char_p = ctypes.c_char_p
_bool = ctypes.c_bool
_int = ctypes.c_int


_lib.BlitzL1_new_sparse_dataset.restype = _pointer
_lib.BlitzL1_new_sparse_dataset.argtypes = [_index_t_p, _index_t_p, _value_t_p, _value_t_p, _size_t, _size_t, _size_t]
_lib.BlitzL1_new_dense_dataset.restype = _pointer
_lib.BlitzL1_new_dense_dataset.argtypes = [_value_t_p, _value_t_p, _size_t, _size_t]
_lib.BlitzL1_get_column_norm.restype = _value_t
_lib.BlitzL1_get_column_norm.argtype = [_pointer, _index_t]
_lib.BlitzL1_get_label_i.restype = _value_t
_lib.BlitzL1_get_label_i.argtype = [_pointer, _index_t]
_lib.BlitzL1_new_solver.restype = _pointer
_lib.BlitzL1_new_solver.argtype = None
_lib.BlitzL1_solve_problem.restype = None
_lib.BlitzL1_solve_problem.argtype = [_pointer, _pointer, _value_t, _char_p, _value_t_p, _value_t, _char_p, _value_t, _value_t, _int, _char_p]
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
_lib.BlitzL1_compute_lambda_max.restype = _value_t
_lib.BlitzL1_compute_lambda_max.argtype = [_pointer, _pointer, _char_p]

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


def data_as(obj, ctypes_type):
  if obj.dtype != ctypes_type:
    obj = obj.astype(ctypes_type)
  return (obj, obj.ctypes.data_as(ctypes_type))


class _L1Problem(object):
  def __init__(self, A, b):
    self._loss_arg = _char_p(self._LOSS_TYPE.encode('utf-8'))
    self._load_dataset(A, b)

  def _load_dataset(self, A, b):
    self._shape = A.shape
    n = _size_t(A.shape[0])
    d = _size_t(A.shape[1])
    (self._b, labels_arg) = data_as(b, _value_t_p)
    if sparse.issparse(A):
      if not sparse.isspmatrix_csc(A):
        A = A.tocsc()
      (self._indices, indices_arg) = data_as(A.indices, _index_t_p)
      (self._indptr, indptr_arg) = data_as(A.indptr, _index_t_p)
      (self._data, data_arg) = data_as(A.data, _value_t_p)
      nnz = _size_t(A.nnz)
      self._dataset = _lib.BlitzL1_new_sparse_dataset(
          indices_arg, indptr_arg, data_arg, labels_arg, n, d, nnz)
    else:
      if not A.flags.f_contiguous:
        A = np.asfortranarray(A)
      (self._data, data_arg) = data_as(A, _value_t_p)
      self._dataset = _lib.BlitzL1_new_dense_dataset(
                              data_arg, labels_arg, n, d)

  def _get_A_column_norm(self, j):
    return _lib.BlitzL1_get_column_norm(self._dataset, _index_t(j))

  def _get_label_i(self, i):
    return _lib.BlitzL1_get_label_i(self._dataset, _index_t(i))

  def compute_lambda_max(self):
    return _lib.BlitzL1_compute_lambda_max(_solver, self._dataset, self._loss_arg)

  def solve(self, 
            l1_penalty, 
            initial_x=None, 
            initial_intercept=None, 
            log_directory=""):

    (n, d) = self._shape

    # Initial conditions:
    if initial_x is not None:
      x = initial_x
    else:
      x = np.zeros(d)
    (x, x_arg) = data_as(x, _value_t_p)
    if initial_intercept is not None:
      intercept_arg = _value_t(initial_intercept)
    else:
      intercept_arg = _value_t(0.0)

    # Regularization strength:
    lambda_arg = _value_t(l1_penalty)

    # Log directory:
    if log_directory:
      try:
        os.mkdir(log_directory)
      except:
        pass
    log_dir_arg = _char_p(log_directory.encode('utf-8'))

    # Misc solution variables:
    obj_arg = _value_t()
    duality_gap_arg = _value_t()
    num_itr_arg = _int()
    solution_status = " " * 64
    solution_status_arg = _char_p(solution_status.encode('utf-8'))

    # Solve problem:
    _lib.BlitzL1_solve_problem(_solver, 
                               self._dataset, 
                               lambda_arg, 
                               self._loss_arg, 
                               x_arg, 
                               ctypes.byref(intercept_arg), 
                               solution_status_arg,
                               ctypes.byref(obj_arg), 
                               ctypes.byref(duality_gap_arg), 
                               ctypes.byref(num_itr_arg),
                               log_dir_arg) 

    solution_status = solution_status.strip().strip('\x00')

    # Return solution object:
    return self._SOLUTION_TYPE(x, 
                               intercept_arg.value, 
                               obj_arg.value, 
                               duality_gap_arg.value, 
                               num_itr_arg.value,
                               solution_status)

def load_solution(filepath):
  in_file = open(filepath)
  sol = pickle.load(in_file)
  in_file.close
  return sol

class _Solution(object):
  def __init__(self, x, intercept, obj, duality_gap, num_itr, status):
    self.x = x
    self.intercept = intercept
    self.objective_value = obj
    self.duality_gap = duality_gap
    self.status = status
    self._num_iterations = num_itr

  def _compute_Ax(self, A):
    if sparse.issparse(A):
      result = A * np.mat(self.x).T + self.intercept
      return np.array(result).flatten()
    else:
      return np.dot(A, self.x) + self.intercept

  def save(self, filepath):
    out_file = open(filepath, "w")
    pickle.dump(self, out_file)
    out_file.close()

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
  _LOSS_TYPE = "squared"
  _SOLUTION_TYPE = LassoSolution

class LogRegProblem(_L1Problem):
  _LOSS_TYPE = "logistic"
  _SOLUTION_TYPE = LogRegSolution


