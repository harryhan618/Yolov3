from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "yolo_to_box",
        ["yolo_to_box.pyx"],
        include_dirs=[np.get_include()],
    )
]

class custom_build_ext(build_ext):
    def build_extensions(self):
        build_ext.build_extensions(self)

setup(
    name='yolo_to_box',
    ext_modules=cythonize(ext_modules, build_dir="build"),
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext}
)