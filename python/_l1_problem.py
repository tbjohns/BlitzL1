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


_lib.BlitzL1_new_sparse_dataset.restype = _pointer
_lib.BlitzL1_new_sparse_dataset.argtypes = [_index_t_p, _size_t_p, _value_t_p, _value_t_p, _index_t, _index_t, _size_t]
_lib.BlitzL1_get_column_norm.restype = _value_t
_lib.BlitzL1_get_column_norm.argtype = [_pointer, _index_t]
_lib.BlitzL1_get_label_i.restype = _value_t
_lib.BlitzL1_get_label_i.argtype = [_pointer, _index_t]
_lib.BlitzL1_new_solver.restype = _pointer
_lib.BlitzL1_new_solver.argtype = None
_lib.BlitzL1_solve_problem.restype = None
_lib.BlitzL1_solve_problem.argtype = [_pointer, _pointer, _value_t, _char_p, _value_t_p, _value_t]

_solver = _lib.BlitzL1_new_solver()

def data_as(obj, ctypes_type, force_return_obj=False):
  return_obj = obj
  if obj.dtype != ctypes_type:
    obj = obj.astype(ctypes_type)
    return_obj = obj
  return (return_obj, obj.ctypes.data_as(ctypes_type))


class _L1Problem:
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

  def solve(self, lam):
    lambda_arg = _value_t(lam)
    (n, d) = self.shape
    (x, x_arg) = data_as(np.zeros(d), _value_t_p)
    intercept_arg = _value_t()
    loss_arg = _char_p(self.loss_type())
    _lib.BlitzL1_solve_problem(_solver, self.dataset, lambda_arg, loss_arg, x_arg, ctypes.byref(intercept_arg))
    print intercept_arg.value
    print x
    

class LassoProblem(_L1Problem):
  def loss_type(self):
    return "squared"


