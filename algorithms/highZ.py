import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self):
        super().__init__()
        self.name = 'High Z'
        self.symbol = 'Y'
        self.active = True
        self.b = -32
        self.w = 0.5
        self.minValue = -32
        self.maxValue = 90

    def process(self, sweep):
        super().process(sweep)

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        letter = 'U'
        d = sweep.products['Z'] + 5.0
        print('     {} = {}'.format(letter, d[0,0:10:]))
        print('         {}'.format(d[1,0:10:]))
        print('         [  ...')
        print('         {}'.format(d[-2,0:10:]))
        print('         {}'.format(d[-1,0:10:]))

        return d
