import radarkit

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Simple V Unfold'
        self.productNames = 'Unfolded Velocity'
        self.units = 'MetersPerSecond'
        self.symbols = 'U'
        self.bs = 0.117647
        self.ws = 4.25
        self.minValue = -30.0
        self.maxValue = 30.0
        self.active = True

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if V does not exist
        if 'V' not in sweep.products:
            radarkit.logger.warning('Product V does not exist.')
            return None

        # Just a simple shift
        d = sweep.products['V'] + 0.5

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol)

        return d
