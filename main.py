#!/usr/local/bin/python

import argparse
import radarkit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main")
    parser.add_argument('-H', '--host', default='localhost', type=str, help='hostname (default localhost)')
    parser.add_argument('-p', '--port', default=10000, type=int, help='port number (default 10000)')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='increases verbosity level')
    args = parser.parse_args()

    # radarkit.test(0, debug=bool(args.debug))

    try:
        radar = radarkit.Radar(ipAddress=args.host, verbose=args.verbose)
        radar.start()
    except KeyboardInterrupt:
        print('')
        radar.stop()
