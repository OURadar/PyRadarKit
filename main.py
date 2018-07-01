#!/usr/local/bin/python

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
                        ' \n'
                        ' e.g., -T0 101 runs the test to build a tuple of dictionaries.\n'
                        ' '.format(radarkit.testHelpText()))
    parser.add_argument('-a', '--algorithm-folder', default='algorithms', type=str,
                        help='Use a different folder for the collection of algorithms (default "algorithms")')
    parser.add_argument('-s', '--streams', default=None, type=str, 
                        help='Overrides the initial streams. In this mode, the algorithms do not get executed.\n'
                        'This mode is primarily used for debugging.\n'
                        'The available streams are:\n'
                        ' z - Reflectivity\n'
                        ' v - Velocity\n'
                        ' w - Width\n'
                        ' d - Differential Reflectivity (ZDR)\n'
                        ' p - Differential Phase (PhiDP)\n'
                        ' r - Cross-correlation Coefficient (RhoHV)\n'
                        ' \n'
                        ' e.g., -sZV sets the radar to receive reflectivity and velocity.\n'
                        ' ')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='increases verbosity level')
    parser.add_argument('values', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.test >= 0:
        ret = radarkit.test(args.test, args.values, verbose=args.verbose)
        if not ret is None:
            print('Test produces a return = {}'.format(ret))
        quit()

    try:
        radar = radarkit.Radar(ipAddress=args.host, streams=args.streams, algorithmFolder=args.algorithm_folder, verbose=args.verbose)
        radar.start()
        radar.wait()
    except KeyboardInterrupt:
        print('')
        radar.stop()
