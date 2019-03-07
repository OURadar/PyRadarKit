import radarkit
import algorithms
import numpy as np

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'X-Band Attenuation Correction'
        self.productCount = 2
        self.productName = ['Corrected Reflectivity', 'Corrected Differential Reflectivity']
        self.symbol = ['Zc', 'Dc']
        self.unit = ['dBZ', 'dB']
        self.cmap = ['Z', 'D']
        self.b = [-32.0, -10.0]
        self.w = [0.5, 0.1]
        self.active = True

    def process(self, sweep):
        super().process(sweep)

        # Generate a warning and return early if Z does not exist
        if not all([x in sweep.products for x in ['Sh', 'Z', 'D', 'P', 'R']]):
            radarkit.logger.warning('Product Z does not exist.')
            return None

        s = sweep.products['Sh']
        z = sweep.products['Z']
        d = sweep.products['D']
        p = sweep.products['P']
        r = sweep.products['R']

        # Call the SCWC algorithm
        #zc, dc = algorithms.scwc(s, z, d, p, r)
        zc, dc = algorithms.zphi(z, d, p)

        # Print something on the screen
        if self.verbose > 0:
            radarkit.showArray(zc, letter=self.symbol[0])
            print('')
            radarkit.showArray(dc, letter=self.symbol[1])

        return zc, dc
