import numpy as N
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0
        self.name = 'Sweep Summary'

    # Every algorithm should have this function defined
    def process(self, sweep):
        print('{}:  N = {}'.format(self.name, sweep.rayCount))

        N.set_printoptions(formatter={'float': '{: 5.1f}'.format})

        for letter in ['Z', 'V', 'W', 'D', 'P', 'R']:
            if letter in sweep.products:
                d = sweep.products[letter]
                print('     {} = {}'.format(letter, d[0,0:10:]))
                print('         {}'.format(d[1,0:10:]))
                print('         [  ...')
                print('         {}'.format(d[-2,0:10:]))
                print('         {}'.format(d[-1,0:10:]))
                print('')
