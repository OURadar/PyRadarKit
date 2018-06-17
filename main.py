#!/usr/local/bin/python

import argparse
import radarkit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-H', '--host', default='localhost', type=str, help='hostname (default localhost)')
    parser.add_argument('-p', '--port', default=10000, type=int, help='port number (default 10000)')
    parser.add_argument('-T', '--test', default=-1, type=int,
                        help='Various tests:\n'
                        '11 - Generating an array.')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='increases verbosity level')
    args = parser.parse_args()

    if args.test >= 0:
        tests = {
            1: lambda x: radarkit.showColors(),
            10: lambda x: print('hello {}'.format(x)),
            11: lambda x: print(radarkit.test(3.14, debug=1))
        }
        if args.test in tests:
            tests.get(args.test)(0)
        else:
            print('Error. Test {} does not exist.'.format(args.test))
        quit()

    try:
        radar = radarkit.Radar(ipAddress=args.host, verbose=args.verbose)
        radar.start()
    except KeyboardInterrupt:
        print('')
        radar.stop()
