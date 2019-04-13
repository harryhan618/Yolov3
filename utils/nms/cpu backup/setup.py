import os.path as osp
from distutils.core import setup, Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# extensions
ext_args = dict(
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args={
        'cc': ['-Wno-unused-function', '-Wno-write-strings'],
    },
)

extensions = [
    Extension('cpu_nms', ['cpu_nms.pyx'], **ext_args),
    Extension('cpu_soft_nms', ['cpu_soft_nms.pyx'], **ext_args),
]



# run the customize_compiler
class custom_build_ext(build_ext):

    def build_extensions(self):
        build_ext.build_extensions(self)


setup(
    name='nms',
    cmdclass={'build_ext': custom_build_ext},
    ext_modules=cythonize(extensions),
)
