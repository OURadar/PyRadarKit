import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self):
        radarkit.Algorithm.__init__(self)
        self.name = 'Algorithm 1'
