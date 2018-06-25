import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self, verbose=0):
        super(main, self).__init__(verbose=verbose)
        self.name = 'Summary'
        self.symbol = 'G'

        # Other variables
        self.threshold = 1.0

    # Every algorithm should have this function implemented
    def process(self, sweep):
        # Call the ancestor method, which shows the sweep summary info
        super(main, self).process(sweep)

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        k = 0
        for letter, dataArray in sweep.products.items():
            if k > 0:
                print('')
            k += 1
            radarkit.showArray(dataArray, letter=letter)
