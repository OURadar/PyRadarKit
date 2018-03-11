PyRadarKit
==========

PyRadarKit is a Python interface to communicate with the RadarKit socket interface. This extension is mainly for parsing the pre-defined structure in RadarKit into well-behaved C arrays and then into NumPy arrays. More coming soon ...


## Requirements

- [RadarKit]
- [Python]
- [NumPy]
- [SciPy]
- [HDF5]
- [NetCDF]

On macOS, if you use brew, the dependencies can be installed via
```shell
brew install numpy scipy --with-python3
``````

## Usage

Launch from command line

```shell
python3 main.py -H localhost
``````

## Other Open Source Projects

- [Baltrad]
- [PyART]
- [wradlib]

[RadarKit]: https://git.arrc.ou.edu/cheo4524/radarkit.git
[Python]: https://www.python.org
[NumPy]: http://www.numpy.org
[SciPy]: https://www.scipy.org
[HDF5]: https://support.hdfgroup.org/HDF5
[NetCDF]: https://www.unidata.ucar.edu/software/netcdf
[Baltrad]: http://theradarcommunity.wikidot.com/tool:2
[PyART]: http://arm-doe.github.io/pyart
[wradlib]: http://wradlib.org


## Some Unfinalized Notes

```shell
export HDF5_DIR=/usr/local/Cellar/hdf5/1.10.1_2
brew install hdf5
```

```shell
pip3 install numpy scipy netcdf
```
