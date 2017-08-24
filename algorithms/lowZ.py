import numpy
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0
        self.name = 'Low Reflectivity'

    # Every algorithm should have this function defined
    def process(self,sweep):
        print('{}:'.format(self.name))
        print('    Nothing yet... just a placeholder')
        #radarkit.showColors()
