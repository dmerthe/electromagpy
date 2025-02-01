# Build C source files for the particles module; mostly used for inspection since
# setup tools builds source files on installation anyway

from Cython.Build import cythonize

cythonize([
    "particles.py"
], annotate=True, compiler_directives={'language_level': "3"}, force=True)
