import radarkit
import time

class main(radarkit.Algorithm):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Z-Shift'
        self.unit = 'dBZ'
        self.symbol = 'Y'
        self.active = True
        self.b = -32
        self.w = 0.5
        self.shiftFactor = 5.0

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if Z does not exist
        if 'Z' not in sweep.products:
            radarkit.logger.warning('Product Z does not exist.')
            return None

        # Just a simple shift
        d = sweep.products['Z'] + self.shiftFactor

        time.sleep(1)

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol)

        return d
