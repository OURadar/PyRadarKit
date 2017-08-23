import numpy
import scipy
import radarkit

class main(object):
    def __init__(self):
        self.threshold = 1.0

    # Every algorithm should have this function defined
    def process(self,sweep):
        print('highZ algorithm:')
        print('    Nothing yet... just a placeholder')

        for letter in ['Zi', 'Vi', 'Wi', 'Di', 'Pi', 'Ri']:
            if letter in sweep.products:
                d = sweep.products[letter]
                print('    {}i = {}'.format(letter, d[0,0:10:]))
                print('         {}'.format(d[1,0:10:]))
                print('         {}'.format(d[2,0:10:]))
                print('           ...')
                print('         {}'.format(d[-3,0:10:]))
                print('         {}'.format(d[-2,0:10:]))
                print('         {}'.format(d[-1,0:10:]))
                print('\n')

# Every algorithm should have this function defined
    def name(self):
        string = 'High Reflectivity'
        return string
