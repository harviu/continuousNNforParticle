from setuptools import setup
from Cython.Build import cythonize

setup(
    name= 'octree',
    ext_modules=cythonize("octree.pyx",compiler_directives={'language_level' : "3"}),
    zip_safe=False,
)