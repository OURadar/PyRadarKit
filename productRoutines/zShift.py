import radarkit
import algorithms
import numpy as np

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'SCWC'
        self.productCount = 2
        self.productName = ['Corrected Reflectivity', 'Corrected Differential Reflectivity']
        self.symbol = ['Y', 'C']
        self.unit = ['dBZ', 'dB']
        self.cmap = ['Z', 'D']
        self.b = [-32.0, -10.0]
        self.w = [0.5, 0.1]
        self.active = True

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if Z does not exist
        if 'Z' not in sweep.products:
            radarkit.logger.warning('Product Z does not exist.')
            return None

        # Just a simple shift
        d = np.copy(sweep.products['Z']) + 0.1
        e = np.copy(d) - 0.05

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(d, letter=self.symbol[0])
            print('')
            radarkit.showArray(e, letter=self.symbol[1])

        return d, e
