import numpy as N
import scipy as S
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0
        self.name = 'Algorithm 2'

    # Every algorithm should have this function defined
    def process(self,sweep):
        print('{}:'.format(self.name))
        print('    Nothing yet... just a placeholder')
        #radarkit.showColors()
