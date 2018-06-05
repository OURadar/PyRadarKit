import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self):
        radarkit.Algorithm.__init__(self)
        self.name = 'Sweep Summary'
        self.active = True

        # Other variables
        self.threshold = 1.0

    # Every algorithm should have this function implemented
    def process(self, sweep):
        print('{}:  N = {}'.format(self.name, sweep.rayCount))
        #radarkit.algorithm.process(self, sweep)

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

        u = sweep.products['Z'] + 5.0

        return u
