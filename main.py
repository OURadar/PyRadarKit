#!/usr/local/bin/python

import os
import sys

MIN_PYTHON = (3, 4)
if sys.version_info < MIN_PYTHON:
    sys.exit('Python %s or later is required.\n' % '.'.join("%s" % n for n in MIN_PYTHON))

import argparse
import radarkit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-H', '--host', default='localhost', type=str, help='hostname (default localhost)')
    parser.add_argument('-p', '--port', default=10000, type=int, help='port number (default 10000)')
    parser.add_argument('-T', '--test', default=-1, type=int,
                        help='Various tests:\n'
                        '{}'
                        ' '.format(radarkit.testHelpText()))
    parser.add_argument('-a', '--product-routines', default='productRoutines', type=str,
                        help='Use a different folder for the collection of product algorithms (default "productRoutines")')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='increases verbosity level')
    parser.add_argument('values', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.test >= 0:
        ret = radarkit.test(args.test, args.values, verbose=args.verbose)
        if not ret is None:
            print('Test produces a return = {}'.format(ret))
        quit()

    try:
        radarkit.showName()
        radar = radarkit.Radar(ipAddress=args.host, productRoutinesFolder=args.product_routines, verbose=args.verbose)
        radar.start()
        radar.wait()
    except KeyboardInterrupt:
        print('')
        radar.stop()
