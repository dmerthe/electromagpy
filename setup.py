import os
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        os.path.join("electromagpy", "fields", "field.py"),
        os.path.join("electromagpy", "fields", "electrostatic.py"),
        os.path.join("electromagpy", "fields", "magnetostatic.py"),

        os.path.join("electromagpy", "particles", "particles.py")
    ], annotate=True, force=True, language='c++')
)
