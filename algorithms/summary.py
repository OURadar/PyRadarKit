import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self):
        super().__init__()
        self.name = 'Summary'
        self.symbol = 'G'

        # Other variables
        self.threshold = 1.0

    # Every algorithm should have this function implemented
    def process(self, sweep):
        # Call the ancestor method, which shows the sweep summary info
        super().process(sweep)

        N.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        for letter in ['Z', 'V', 'W', 'D', 'P', 'R']:
            if letter in sweep.products:
                d = sweep.products[letter]
                radarkit.showArray(d, letter=letter)
