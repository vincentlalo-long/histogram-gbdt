import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith("linux"):
    extra_compile_args.extend(["-O3", "-fopenmp"])
    extra_link_args.append("-fopenmp")
elif sys.platform == "darwin":
    extra_compile_args.extend(["-O3", "-Xpreprocessor", "-fopenmp"])
    extra_link_args.extend(["-lomp"])
elif sys.platform == "win32":
    extra_compile_args.extend(["/O2", "/openmp"])

ext_modules = [
    Pybind11Extension(
        "gbdt._core",
        ["src/gbdt/_core.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
