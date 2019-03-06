import radarkit

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'Stencil'
