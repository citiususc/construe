from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from Cython.Distutils import build_ext
import numpy as np

py_inc = [get_python_inc()]
np_inc = [np.get_include()]

ext_modules = [Extension("dtw",
                         ["./cdtw.c",
                          "./dtw.pyx"],
                         include_dirs=py_inc + np_inc,
                         libraries=["m"])]

cmdclass = {"build_ext": build_ext}
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass
)
