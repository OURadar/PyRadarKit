import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self, verbose=0):
        super(main, self).__init__(verbose=verbose)
        self.name = 'Z-Shift'
        self.symbol = 'Y'
        self.unit = 'dBZ'
        self.active = True
        self.b = -32
        self.w = 0.5
        self.minValue = -32
        self.maxValue = 90
        self.shiftFactor = 5.0

    def process(self, sweep):
        super(main, self).process(sweep)

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        if 'Z' not in sweep.products:
            radarkit.logger.warning('Product Z does not exist.')
            return None

        # Just a simple shift
        d = sweep.products['Z'] + self.shiftFactor

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol)

        return d
