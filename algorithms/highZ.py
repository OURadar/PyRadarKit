import numpy
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0

    # Every algorithm should have this function defined
    def process(self,payload):
        print('highZ algorithm:')
        ray = radarkit.test(payload)
        print('    PyRadarKit: EL {0:0.2f} deg   AZ {1:0.2f} deg'.format(ray['elevation'], ray['azimuth']), end='')
        print('   Zi = {}'.format(ray['data'][0:10:]))

    # Every algorithm should have this function defined
    def name(self):
        string = 'High Reflectivity'
        return string
