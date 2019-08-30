import radarkit

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Z Shift'
        self.productName = 'Shifted Reflectivity'
        self.productCount = 1
        self.symbol = 'Y'
        self.unit = 'dBZ'
        self.cmap = 'Z'
        self.b = 0.5
        self.w = -32
        self.minValue = -20.0
        self.maxValue = 80.0

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if V does not exist
        if 'Z' not in sweep.products:
            radarkit.logger.warning('Product Z does not exist.')
            return None

        # Just a simple shift
        y = sweep.products['Z'] + 0.5

        # Print something on the screen
        if self.verbose > 1:
            radarkit.showArray(y, letter=self.symbol)

        return y
