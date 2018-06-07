import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self):
        super().__init__()
        self.name = 'Sweep Summary'

        # Other variables
        self.threshold = 1.0

    # Every algorithm should have this function implemented
    def process(self, sweep):
        # Call the ancestor method, which shows the sweep summary info
        super().process(sweep)

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        for letter in ['Z', 'V', 'W', 'D', 'P', 'R']:
            if letter in sweep.products:
                d = sweep.products[letter]
                print('     {} = {}'.format(letter, d[0,0:10:]))
                print('         {}'.format(d[1,0:10:]))
                print('         [  ...')
                print('         {}'.format(d[-2,0:10:]))
                print('         {}'.format(d[-1,0:10:]))
                print('')
