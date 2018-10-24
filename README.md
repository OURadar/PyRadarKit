PyRadarKit
==========

PyRadarKit is a Python interface to communicate with the RadarKit socket interface. The main key feature is to bridge the base moment data from the RadarKit framework to a Python processing space, which allows for array processing and simple algorithm development. New product may be requested to be sent back to RadarKit for arhiving, live distribution, etc., which are all part of the RadarKit work flow. Another key feature is parsing the pre-defined structure in RadarKit into C arrays and then to NumPy arrays.

It also contains extensions that utilizes [RadarKit] to perform some low-level tasks, such as reading the native product files.


## Requirements

- [RadarKit] 1.2.10 or later
- [Python] 3.3 or later
- [NumPy] 1.11.0 or later
- [SciPy]

On macOS, I recommend [Homebrew] package manager, numpy and scipy can be installed via
```shell
brew install numpy scipy
``````

To build the extension, use the included Makefile:
```shell
make
```

Someitmes you might need to clean the current built to force a re-build, which can be done as:
```shell
make clean && make
```

## Usage

Launch from command line

```shell
main.py -H localhost
``````

The built-in help text is usually up-to-date

```
usage: main [-h] [-H HOST] [-p PORT] [-T TEST] [-s STREAMS] [-v] ...

positional arguments:
  values

optional arguments:
  -h, --help            show this help message and exit
  -H HOST, --host HOST  hostname (default localhost)
  -p PORT, --port PORT  port number (default 10000)
  -T TEST, --test TEST  Various tests:
                         0 - Extension module sub-tests:
                              - 100 - Building a simple value.
                              - 101 - Building a tuple of two dictionaries.
                         1 - Show color output from RadarKit.
                        11 - Generating an array.
                         
                         e.g., -T0 101 runs the test to build a tuple of dictionaries.
                         
  -s STREAMS, --streams STREAMS
                        Overrides the initial streams. In this mode, the algorithms do not get executed.
                        This mode is primarily used for debugging.
                        The available streams are:
                         z - Reflectivity
                         v - Velocity
                         w - Width
                         d - Differential Reflectivity (ZDR)
                         p - Differential Phase (PhiDP)
                         r - Cross-correlation Coefficient (RhoHV)
                         
                         e.g., -sZV sets the radar to receive reflectivity and velocity.
                         
  -v, --verbose         increases verbosity level
```

## Developing Your Own Algorithms

Each algorithm must be its own Python script under the folder algorithms. Here are the requirements:
- The algorithm must be a class of itself, which is derived as a subclass from `radarkit.Algorithm`.
- Each algorithm is allowed to return one product only, which must be the same size as Z.
- The method `process()` must be overriden to generate and return the product.

A trivial algorithm is provided as a `zShift.py`, which can be found under the algorithms folder. Here's an excerpt of the example to illustrate the fundamental concepts. This algorithm uses the base moment 'Z' and add an offset value to it to produce a new product 'Y'.

```python
class main(radarkit.Algorithm):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Z-Shift'
        self.unit = 'dBZ'
        self.symbol = 'Y'
        self.active = True
        self.b = -32
        self.w = 0.5
        self.shiftFactor = 5.0

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if Z does not exist
        if 'Z' not in sweep.products:
            radarkit.logger.warning('Product Z does not exist.')
            return None

        # Just a simple shift
        d = sweep.products['Z'] + self.shiftFactor

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol)

        return d
```


## Other Open Source Projects

- [Baltrad]
- [PyART]
- [wradlib]

[Homebrew]: https://brew.sh
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
pip3 install numpy scipy netcdf
```
