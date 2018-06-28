import os
import sys

from . import rk
from .misc import *

def test(number, args=None, debug=False):
    tests = {
        0: lambda x: rk.test(args, debug=1),
        1: lambda x: print(args),
        10: lambda x: showName()
    }
    if number in tests:
        print('args for sub-module = {}'.format(args))
        return tests.get(number)(args)
    else:
        print('Error. Test {} does not exist.'.format(number))
