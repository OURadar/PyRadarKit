import sys
import logging
import signal
import socket
import time
import threading
import struct

from argparse import ArgumentParser

import radarkit

sys.path.insert(0, 'algorithms')

if __name__ == "__main__":
    parser = ArgumentParser(prog="main")
    parser.add_argument('-H', '--host', default='localhost', help='hostname (default localhost)')
    parser.add_argument('-p', '--port', default=10000, type=int, help='port number (default 10000)')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='increases verbosity level')
    args = parser.parse_args()

    # print('Version {}'.format(sys.version_info))

    # radarkit.test(0, debug=bool(args.debug))

    try:
        radar = radarkit.Radar(ipAddress=args.host, verbose=args.verbose)
        radar.start()
    except KeyboardInterrupt:
        print('')
        print('Deactivating radar ...')
        radar.stop()
        print('Done')
