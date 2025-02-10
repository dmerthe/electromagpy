import os
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.fast_fail = True

setup(
    ext_modules=cythonize([
        os.path.join("electromagpy", "fields", "field.py"),
        os.path.join("electromagpy", "fields", "electrostatic.py"),
        os.path.join("electromagpy", "fields", "magnetostatic.py"),

        os.path.join("electromagpy", "particles", "particles.py")
    ], annotate=True, force=True, compiler_directives={"language_level": 3})
)
