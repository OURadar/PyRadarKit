import numpy
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0
        radarkit.init()

    # Every algorithm should have this function defined
    def process(self,sweep):
        print('lowZ algorithm:')
        print('    Nothing yet... just a placeholder')
        #radarkit.showColors()

    # Every algorithm should have this function defined
    def name(self):
        string = 'Low Reflectivity'
        return string
