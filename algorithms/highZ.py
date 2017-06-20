import numpy
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0

    # Every algorithm should have this function defined
    def process(self,payload):
        print('highZ algorithm:')
        d = radarkit.test(payload)
        print('    PyRadarKit: Zi = {}'.format(d[0:10:]))

    # Every algorithm should have this function defined
    def name(self):
        string = 'High Reflectivity'
        return string
