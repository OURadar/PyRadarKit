from setuptools import setup, Extension

import os
import numpy.distutils.misc_util

# Define the extension module
inc_dirs = ['/usr/local/include'] + numpy.distutils.misc_util.get_numpy_include_dirs()
lib_dirs = ['/usr/local/lib']

print('\033[34m===>\033[0m inc_dirs = {}'.format(inc_dirs))
print('\033[34m===>\033[0m lib_dirs = {}'.format(lib_dirs))

install_requires = [
    'numpy',
    'scipy',
    'matplotlib'
]
scripts = [
    '__init__.py',
    'misc.py'
]
console_scripts = [
    'gui=gui.__main__:main'
]
gui_scripts = []

rk = Extension('radarkit.rk',
               ['radarkit/rk.c'],
               include_dirs=inc_dirs,
               library_dirs=lib_dirs,
               libraries=['radarkit', 'fftw3f', 'netcdf'],
               extra_compile_args=['-Wno-strict-prototypes', '-Wno-microsoft'])

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='PyRadarKit',
    version='0.5',
    description='The Python Extension of RadarKit.',
    author='Boonleng Cheong',
    author_email='boonleng@ou.edu',
    url='https://github.com/ouradar/pyradarkit',
    package_dir={'radarkit': 'radarkit'},
    packages=['radarkit'],
    ext_modules=[rk],
    install_requires=install_requires,
    zip_safe=False,
    license='MIT'
)
