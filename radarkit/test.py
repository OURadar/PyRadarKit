import os
import sys

from . import rk
from .misc import *

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
        '''.format(rk.testByNumberHelp())
    return text
