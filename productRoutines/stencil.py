import radarkit

class main(radarkit.ProductRoutine):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.name = 'A Stencil Product Routine'

    # Every algorithm should have this function implemented
    def process(self, sweep):
        super().process(sweep)
        return None
