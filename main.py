#!/usr/local/bin/python

import os
import sys

MIN_PYTHON = (3, 4)
if sys.version_info < MIN_PYTHON:
    sys.exit('Python %s or later is required.\n' % '.'.join("%s" % n for n in MIN_PYTHON))

import argparse
import radarkit

def showName():
    rows, columns = os.popen('stty size', 'r').read().split()
    c = int(columns)
    print('Version {}\n'.format(sys.version_info))
    #print(radarkit.colorize('{}\n{}\n{}'.format(' ' * c, 'Algorithm Manager'.center(c, ' '), ' ' * c), "\033[38;5;15;48;5;28m"))
    print(radarkit.colorize('{}\n{}\n{}'.format(' ' * c, 'Algorithm Manager'.center(c, ' '), ' ' * c), "\033[38;5;15;48;5;241m"))
    print(radarkit.colorize('{}\n{}\n{}'.format(' ' * c, 'PyRadarKit {}'.format(radarkit.version_info).center(c, ' '), ' ' * c), radarkit.COLOR.python))
    print(radarkit.colorize('{}\n{}\n{}'.format(' ' * c, 'RadarKit {}'.format(radarkit.rk.version()).center(c, ' '), ' ' * c), radarkit.COLOR.radarkit))
    print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-H', '--host', default='localhost', type=str, help='hostname (default localhost)')
    parser.add_argument('-p', '--port', default=10000, type=int, help='port number (default 10000)')
    parser.add_argument('-T', '--test', default=-1, type=int,
                        help='Various tests:\n'
                        '{}'
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
        showName()
        radar = radarkit.Radar(ipAddress=args.host, streams=args.streams, algorithmFolder=args.algorithm_folder, verbose=args.verbose)
        radar.start()
        radar.wait()
    except KeyboardInterrupt:
        print('')
        radar.stop()
