import distutils
from distutils.core import setup, Extension
import os

source_list = ["src/%s" % f for f in os.listdir("src") 
                            if f.endswith("cpp")]

ext = Extension("blitzl1.libblitzl1",
                include_dirs = ["src"],
                extra_compile_args=['-O3'],
                sources = source_list)

setup(name = "BlitzL1",
      version = "0.0",
      description = "Fast, principled, L1-regularized loss minimization",
      package_dir = {"blitzl1": "python"},
      packages = ["blitzl1"],
      ext_modules = [ext])
