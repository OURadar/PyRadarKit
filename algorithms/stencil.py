import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Stencil'
