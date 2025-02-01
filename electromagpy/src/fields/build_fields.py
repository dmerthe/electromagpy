# Build C source files for the fields module; mostly used for inspection since
# setup tools builds source files on installation anyway

from Cython.Build import cythonize

cythonize([
    "field.py",
    "electrostatic.py"
], annotate=True, compiler_directives={'language_level': "3"}, force=True)
