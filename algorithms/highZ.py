import numpy as N
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0
        self.name = 'High Reflectivity'

    # Every algorithm should have this function defined
    def process(self,sweep):
        print('highZ algorithm:')
        print('    Nothing yet... just a placeholder')
