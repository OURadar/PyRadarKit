import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'V-Unfold'
        self.symbol = 'U'
        self.unit = 'MetersPerSecond'
        self.active = True
        self.b = 0.117647
        self.w = 4.25
        self.minValue = -30.0
        self.maxValue = 30.0

    def process(self, sweep):
        super().process(sweep)

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        if 'V' not in sweep.products:
            radarkit.logger.warning('Product V does not exist.')
            return None

        # Just a simple shift
        d = sweep.products['V'] + 0.5

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol)

        return d
