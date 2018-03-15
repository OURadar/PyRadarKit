from distutils.core import setup, Extension
import numpy.distutils.misc_util

# Define the extension module
inc_dirs = ['/usr/local/include'] + numpy.distutils.misc_util.get_numpy_include_dirs()
lib_dirs = ['/usr/local/lib']

print('\033[34m===>\033[0m inc_dirs = {}'.format(inc_dirs))
print('\033[34m===>\033[0m lib_dirs = {}'.format(lib_dirs))

rkstruct = Extension('rkstruct', sources=['rkstruct.c'],
                     include_dirs = inc_dirs,
                     library_dirs = lib_dirs,
                     extra_compile_args = ['-Wno-strict-prototypes', '-Wno-microsoft'],
                     libraries = ['radarkit', 'fftw3f', 'netcdf'])

# Run the setup
setup(
      name = 'PyRK',
      version = '0.2',
      description = 'RadarKit Python Extension',
      ext_modules = [rkstruct]
)
