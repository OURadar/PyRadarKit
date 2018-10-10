import os
import sys
import textwrap
import pkg_resources

MIN_PYTHON = (3, 4)
if sys.version_info < MIN_PYTHON:
    sys.exit('Python %s or later is required.\n' % '.'.join("%s" % n for n in MIN_PYTHON))

class COLOR:
    reset = "\033[0m"
    red = "\033[38;5;196m"
    orange = "\033[38;5;214m"
    yellow = "\033[38;5;226m"
    lime = "\033[38;5;118m"
    green = "\033[38;5;46m"
    teal = "\033[38;5;49m"
    iceblue = "\033[38;5;51m"
    skyblue = "\033[38;5;45m"
    blue = "\033[38;5;27m"
    purple = "\033[38;5;99m"
    indigo = "\033[38;5;201m"
    hotpink = "\033[38;5;199m"
    pink = "\033[38;5;213m"
    deeppink = "\033[38;5;198m"
    salmon = "\033[38;5;210m"
    python = "\033[38;5;226;48;5;24m"
    radarkit = "\033[38;5;15;48;5;124m"

def colorize(string, color):
    return '{}{}{}'.format(color, string, COLOR.reset)

print('Using version {} ...'.format(colorize(sys.version_info, COLOR.lime)))

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

if not is_installed('scipy>=0.19.0'):
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

print('{} inc_dirs = {}'.format(colorize('\033[34m===>\033[0m', COLOR.orange), inc_dirs))
print('{} inc_libs = {}'.format(colorize('\033[34m===>\033[0m', COLOR.purple), lib_dirs))

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
               extra_compile_args=['-std=gnu99', '-Wno-strict-prototypes'])

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
