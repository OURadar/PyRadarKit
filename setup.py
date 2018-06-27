from __future__ import print_function

import os
import sys
import textwrap
import pkg_resources

print('Using version {} ...'.format(sys.version_info))

def is_installed(requirement):
    try:
        print('Checking for {} ...'.format(requirement))
        pkg_resources.require(requirement)
    except pkg_resources.ResolutionError:
        print('{} not found'.format(requirement))
        return False
    else:
        result = pkg_resources.require(requirement)
        print('{} found {}'.format(requirement, result[0]))
        return True

if not is_installed('numpy>=1.11.0'):
    print(textwrap.dedent("""
        Error: numpy needs to be installed first. You can install it via:

        $ yum install scipy

        or
        $ pip install numpy
        $ pip3 install numpy
        """), file=sys.stderr)
    exit(1)

if not is_installed('scipy>=1.0.0'):
    print(textwrap.dedent("""
        Error: scipy needs to be installed first. You can install it via:

        $ yum install scipy

        or
        
        $ pip install scipy
        $ pip3 install scipy
        """), file=sys.stderr)
    exit(1)

# if not is_installed('python-devel'):
#     print(textwrap.dedent("""
#             Error: python-devel needs to be installed first. You can install it via:

#             $ yum install python-devel
#             $ yum install python3-devel
#             """), file=sys.stderr)
#     exit(1)

from setuptools import setup, Extension
import numpy.distutils.misc_util

# Define the extension module
inc_dirs = ['/usr/local/include'] + numpy.distutils.misc_util.get_numpy_include_dirs()
lib_dirs = ['/usr/local/lib']

print('\033[34m===>\033[0m inc_dirs = {}'.format(inc_dirs))
print('\033[34m===>\033[0m lib_dirs = {}'.format(lib_dirs))

install_requires = [
    'enum',
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
               extra_compile_args=['-std=gnu99', '-Wno-strict-prototypes', '-Wno-microsoft'])

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
