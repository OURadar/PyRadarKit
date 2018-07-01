import os
import sys

from . import rk
from .misc import *

'''
    Tests are divided into several levels:
    Level 0 - Tests 0 - 99
            - Straigh from RadarKit
   
    Level 1 - Tests 100 - 199
            - From C extension module of PyRadarKit
   
    Level 2 - Tests 200 - 299
            - From Python space of PyRadarKit
'''
def test(number, args=None, debug=False):
    tests = {
        200: lambda x: showName(),
        201: lambda x: print('x = {}   args = {}'.format(x, args))
    }
    if number < 200:
        print('args for testByNumber = {}'.format((number, *args)))
        return rk.testByNumber((number, *args))
    elif number in tests:
        print('args for sub-module = {}'.format(args))
        return tests.get(number)(args)
    else:
        print('Error. Test {} does not exist.'.format(number))
        return None

def testHelpText():
    text = '''{}
        100 - Building a simple value.
        101 - Building a tuple of two dictionaries.
        
        200 - Show framework header
        201 - Show input arguments
        '''.format(rk.testByNumberHelp())
    return text
