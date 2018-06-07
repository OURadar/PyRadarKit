import numpy as N
import scipy as S
import radarkit

class main(radarkit.Algorithm):
    def __init__(self):
        super().__init__()
        self.name = 'Algorithm 2'
