from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_module = [Extension("fitramp_movingsource", 
                       ['fitramp_movingsource.pyx'],
                        extra_compile_args=['-funsafe-math-optimizations', '-fno-rounding-math', '-O3',], 
)]

setup(
    name = 'fitramp_movingsource',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_module),
)

