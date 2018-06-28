import os
import sys

from . import rk
from .misc import *

def test(number, args=None, debug=False):
    tests = {
        1: lambda x: rk.showColors(),
        10: lambda x: showName(),
        11: lambda x: print(rk.test(3.14, debug=1))
    }
    if number in tests:
        return tests.get(number)(args)
    else:
        print('Error. Test {} does not exist.'.format(number))
