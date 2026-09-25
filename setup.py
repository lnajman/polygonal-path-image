"""Build the Cython extension; project metadata lives in pyproject.toml."""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "polygonal_path_image._core",
                ["src/polygonal_path_image/_core.pyx"],
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ],
        compiler_directives={"language_level": 3, "embedsignature": True},
        build_dir="build/cython",
    )
)
