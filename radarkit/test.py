import os
import sys

from . import rk
from .misc import *

def test(number, args=None, debug=False):
    tests = {
        0: lambda x: rk.test(args, debug=1),
        1: lambda x: print(args),
        2: lambda x: rk.showColors(),
        10: lambda x: showName()
    }
    print('args for sub-module = {}'.format(args))
    if number in tests:
        return tests.get(number)(args)
    else:
        print('Error. Test {} does not exist.'.format(number))
