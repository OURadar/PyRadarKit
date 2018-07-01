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
def test(number, args=None, debug=False, verbose=0):
    tests = {
        200: lambda x: showName(),
        201: lambda x: print('x = {}   args = {}'.format(x, args))
    }
    if number >= 0 and number < 200:
        if verbose > 0:
            print('args for testByNumber = {}'.format(args))
        return rk.testByNumber(number, args=args, verbose=verbose)
    elif number in tests:
        if verbose > 0:
            print('args for sub-module = {}'.format(args))
        return tests.get(number)(args)
    else:
        print('Error. Test {} does not exist.'.format(number))
        return None

def testHelpText():
    text = '''
        RadarKit
        --------
{}
        
        C-Ext Module of PyRadarKit
        --------------------------
        100 - Test retrieving the RadarKit framework version through PyRadarKit
        101 - Building an integer PyObject with only one integer value.
        102 - Building a tuple PyObject that contains two dictionaries.
        
        
        Python Space of PyRadarKit
        --------------------------
        200 - Test showing framework header
        201 - Test receiving additional input arguments as a list
        
e.g., -T102 runs the test to build a tuple of dictionaries
        '''.format(rk.testByNumberHelp())
    return text
