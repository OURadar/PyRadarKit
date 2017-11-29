PyRadarKit
==========

PyRadarKit is a Python interface to communicate with the RadarKit socket interface. This extension is mainly for parsing the pre-defined structure in RadarKit into well-behaved C arrays and then into NumPy arrays. More coming soon ...


## Requirements

- [RadarKit]
- [Python]
- [NumPy]
- [SciPy]

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
[Baltrad]: http://theradarcommunity.wikidot.com/tool:2
[PyART]: http://arm-doe.github.io/pyart
[wradlib]: http://wradlib.org
