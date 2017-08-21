import numpy
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0

    # Every algorithm should have this function defined
    def process(self,sweep):
        print('highZ algorithm:')
        print('    Nothing yet... just a placeholder')

        z = sweep.products['Z']

        print('   Zi = {}'.format(z[0,0:10:]))
        print('      = {}'.format(z[1,0:10:]))
        print('      = {}'.format(z[-1,0:10:]))

    # Every algorithm should have this function defined
    def name(self):
        string = 'High Reflectivity'
        return string
