PyRadarKit
==========

PyRadarKit is a Python interface to communicate with the RadarKit socket interface. The main key feature is to bridge the base moment data from the RadarKit framework to a Python processing space, which allows for array processing and simple product routine development. New products may be requested to be sent back to RadarKit for arhiving, live distribution, etc., which are all part of the RadarKit workflow. Another key feature is parsing the pre-defined structure in RadarKit into C arrays and then to NumPy arrays.

PyRadarKit also comes with a number of notebooks to illustrate the usage of some standalone tools such as product algorithms and image generator.

PyRadarKit also contains extensions that utilizes [RadarKit] to perform some low-level tasks, such as reading the native product files.


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
usage: main [-h] [-H HOST] [-p PORT] [-T TEST] [-a PRODUCT_ROUTINES] [-v] ...

positional arguments:
  values

optional arguments:
  -h, --help            show this help message and exit
  -H HOST, --host HOST  hostname (default localhost)
  -p PORT, --port PORT  port number (default 10000)
  -T TEST, --test TEST  Various tests:
                        
                        RadarKit
                        --------
                         0 - Show types
                         1 - Show colors
                         2 - Test pretty strings
                         3 - Test modulo math macros
                         4 - Test parsing comma delimited values
                         5 - Test parsing values in a JSON string
                         6 - Test initializing a file maanger - RKFileManagerInit()
                         7 - Test reading a preference file - RKPreferenceInit()
                         8 - Test counting files using RKCountFilesInPath()
                         9 - Test the file monitor module - RKFileMonitor()
                        10 - Test the internet monitor module - RKHostMonitorInit()
                        11 - Test initializing a radar system - RKRadarInit()
                        12 - Test converting a temperature reading to status
                        13 - Test getting a country name from position
                        14 - Test generating text for buffer overview
                        15 - Test reading a netcdf file using RKSweepRead(); -T15 FILE
                        16 - Test reading a netcdf file using RKProductRead()
                        17 - Test reading using RKProductCollectionInitWithFilename()
                        
                        20 - SIMD quick test
                        21 - SIMD test with numbers shown
                        22 - Show window types
                        23 - Hilbert transform
                        24 - Optimize FFT performance and generate an fft-wisdom file
                        25 - Show ring filter coefficients
                        
                        30 - Make a frequency hopping sequence
                        31 - Make a TFM waveform
                        32 - Generate a waveform file
                        33 - Test waveform down-sampling
                        34 - Test showing built-in waveform properties
                        35 - Test showing waveform properties; -T35 WAVEFORM_FILE
                        
                        40 - Pulse compression using simple cases
                        41 - Calculating one ray using the Pulse Pair method
                        42 - Calculating one ray using the Pulse Pair Hop method
                        43 - Calculating one ray using the Multi-Lag method with L = 2
                        44 - Calculating one ray using the Multt-Lag method with L = 3
                        45 - Calculating one ray using the Multi-Lag method with L = 4
                        46 - Calculating one ray using the Spectral Moment method
                        
                        50 - Measure the speed of SIMD calculations
                        51 - Measure the speed of pulse compression
                        52 - Measure the speed of various moment methods
                        53 - Measure the speed of cached write
                        
                        
                        C-Ext Module of PyRadarKit
                        --------------------------
                        100 - Test retrieving the RadarKit framework version through PyRadarKit
                        101 - Building an integer PyObject with only one integer value.
                        102 - Building a tuple PyObject that contains two dictionaries.
                            
                            
                        Python Space of PyRadarKit
                        --------------------------
                        200 - Test showing framework header
                        201 - Test receiving additional input arguments as a list
                        
                        e.g., -T102 runs the test to build a tuple of dictionaries
                             
  -a PRODUCT_ROUTINES, --product-routines PRODUCT_ROUTINES
                        Use a different folder for the collection of product algorithms (default "productRoutines")
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
