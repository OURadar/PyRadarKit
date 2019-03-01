import radarkit
import numpy as N

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Summary'
        self.symbol = 'G'

        # Other internal variables
        self.threshold = 1.0

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

    # Every algorithm should have this function implemented
    def process(self, sweep):
        if self.verbose < 2:
            return
        k = 0
        for letter, dataArray in sweep.products.items():
            if k > 0:
                print('')
            k += 1
            radarkit.showArray(dataArray, letter=letter)
