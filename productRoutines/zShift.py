import radarkit
import algorithms

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'SCWC'
        self.productCount = 2
        self.productNames = ['Corrected Reflectivity', 'Corrected Differential Reflectivity']
        self.symbols = ['Y', 'C']
        self.units = ['dBZ', 'dB']
        self.cmaps = ['Z', 'D']
        self.bs = [-32.0, -10.0]
        self.ws = [0.5, 0.1]
        self.active = True

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if Z does not exist
        if 'Z' not in sweep.products:
            radarkit.logger.warning('Product Z does not exist.')
            return None

        # Just a simple shift
        d = sweep.products['Z']

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol)

        return d
