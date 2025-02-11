import os
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.fast_fail = True

extensions = [
    Extension(
        "electromagpy.fields.field",
        [os.path.join("electromagpy", "fields", "field.py")],
        language="c++",
        extra_compile_args=["-std=c++20"]
    ),
    Extension(
        "electromagpy.fields.electrostatic",
        [os.path.join("electromagpy", "fields", "electrostatic.py")],
        language="c++",
        extra_compile_args=["-std=c++20"]
    ),
    Extension(
        "electromagpy.fields.magnetostatic",
        [os.path.join("electromagpy", "fields", "magnetostatic.py")],
        language="c++",
        extra_compile_args=["-std=c++20"]
    ),

    Extension(
        "electromagpy.particles.particles",
        [os.path.join("electromagpy", "particles", "particles.py")],
        language="c++",
        extra_compile_args=["-std=c++20"]
    ),
]

# setup(
#     ext_modules=cythonize([
#         os.path.join("electromagpy", "fields", "field.py"),
#         os.path.join("electromagpy", "fields", "electrostatic.py"),
#         os.path.join("electromagpy", "fields", "magnetostatic.py"),
#
#         os.path.join("electromagpy", "particles", "particles.py")
#     ], annotate=True, force=True, compiler_directives={"language_level": 3})
# )

setup(
    ext_modules=cythonize(
        extensions, annotate=True, force=True,
        compiler_directives={"language_level": 3, "cdivision": True}
    )
)
