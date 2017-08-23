"""
    Python wrapper for C functions to interact with RadarKit
"""

import logging
import math
import socket
import struct
import os
import sys
import glob
import time
import enum
import numpy as N

import rkstruct

logger = logging.getLogger(__name__)

class CONSTANTS:
    IP = '127.0.0.1'
    PORT = 10000
    MAX_GATES = 4096
    BUFFER_SIZE = 262144
    PACKET_DELIM_SIZE = 16

class NETWORK_PACKET_TYPE:
    BYTES = 0
    BEACON = 1
    PLAIN_TEXT = 2
    PULSE_DATA = 3
    RAY_DATA = 4
    HEALTH = 5
    CONTROLS = 6
    COMMAND_RESPONSE = 7
    RADAR_DESCRIPTION = 8
    PROCESSOR_STATUS = 9
    RAY_DISPLAY = 109
    ALERT_MESSAGE = 110
    CONFIG = 111

# Each delimiter has 16-bit type, 16-bit subtype, 32-bit raw size, 32-bit decoded size and 32-bit padding
RKNetDelimiter = b'HHIII'

# Generic functions
def test(payload, debug=False):
    print('debug = {}'.format(debug))
    return rkstruct.test(payload, debug=debug)

def init():
    rkstruct.init()

def showColors():
    rkstruct.showColors()

class Sweep(object):
    """
        An object that encapsulate a sweep
    """
    def __init__(self):
        self.azimuth = 0.0
        self.elevation = 0.0
        self.sweepType = "PPI"
        self.products = {
            'Zi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            'Vi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            'Wi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            'Di': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            'Pi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            'Ri': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            'Z': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.float),
            'V': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.float),
            'W': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.float),
            'D': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.float),
            'P': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.float),
            'R': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.float)
        }

# Radar class
class Radar(object):
    """
        Handles the connection to the radar (created by RadarKit)
        This class allows to retrieval of base data from the radar
    """
    def __init__(self, ipAddress=CONSTANTS.IP, port=CONSTANTS.PORT, timeout=2, verbose=0):
        self.ipAddress = ipAddress
        self.port = port
        self.timeout = timeout
        self.verbose = verbose
        self.netDelimiter = bytearray(CONSTANTS.PACKET_DELIM_SIZE)
        self.payload = bytearray(CONSTANTS.BUFFER_SIZE)
        self.latestPayloadType = 0

        # Initialize an empty list of algorithms
        self.algorithms = []
        self.sweep = Sweep()

    def _recv(self):
        try:
            length = self.socket.recv_into(self.netDelimiter, CONSTANTS.PACKET_DELIM_SIZE)
            if length != CONSTANTS.PACKET_DELIM_SIZE:
                raise ValueError('Length should be {}, not {}'.format(CONSTANTS.PACKET_DELIM_SIZE, length))
            delimiter = struct.unpack(RKNetDelimiter, self.netDelimiter)

            # 1st component: 16-bit type
            # 3rd component: 16-bit subtype
            payloadType = delimiter[0]
            payloadSize = delimiter[2]
            self.latestPayloadType = payloadType

            if payloadType != NETWORK_PACKET_TYPE.BEACON and payloadType != NETWORK_PACKET_TYPE.RAY_DISPLAY:
                print('Delimiter type {} of size {}'.format(payloadType, payloadSize))

            if payloadSize > 0:
                anchor = memoryview(self.payload)
                k = 0
                toRead = payloadSize
                while toRead:
                    length = self.socket.recv_into(anchor, toRead)
                    anchor = anchor[length:]
                    toRead -= length
                    k += 1

        except (socket.timeout, ValueError) as e:
            logger.exception(e)
            raise OSError('Couldn\'t retrieve socket data')

    def start(self):
        self.active = True

        # Loop through all the files under 'algorithms' folder
        print('Loading algorithms ...\n')
        self.algorithmObjects = []
        for script in glob.glob('algorithms/*.py'):
            basename = os.path.basename(script)[:-3]
            mod = __import__(basename)
            obj = getattr(mod, 'main')()
            self.algorithmObjects.append(obj)
            print('\033[38;5;220m{}\033[0m -> {} -> {}'.format(script, basename, obj.name()))

        self.reconnect()

    def stop(self):
        self.active = False

    def reconnect(self):

        while self.active:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            try:
                print('\nConnecting {}:{}...'.format(self.ipAddress, self.port))
                self.socket.connect((self.ipAddress, self.port))
            except:
                t = 3
                while t > 0:
                    print('Retry in {} seconds ...\r'.format(t))
                    time.sleep(1)
                    t -= 1
                self.socket.close()
                continue

            # Request status, Z and V
            #self.socket.send(b'szvwdpr\r\n')
            self.socket.send(b'sZVWDPR\r\n')

            while self.active:
                self._recv()
                if self.latestPayloadType == NETWORK_PACKET_TYPE.RAY_DISPLAY:
                    # Parse the ray
                    ray = rkstruct.parse(self.payload, verbose=self.verbose)
                    # Gather the ray into a sweep
                    ii = int(ray['azimuth'])
                    ng = min(ray['gateCount'], CONSTANTS.MAX_GATES)
                    if self.verbose > 1:
                        print('   \033[38;5;226;48;5;24m PyRadarKit \033[0m \033[38;5;226mEL {0:0.2f} deg   AZ {1:0.2f} deg\033[0m -> {2} / {3}'.format(ray['elevation'], ray['azimuth'], ii, ray['sweepEnd']))
                        N.set_printoptions(formatter={'float': '{: 5.1f}'.format})
                        for letter in ['Z', 'V', 'W', 'D', 'P', 'R']:
                            if letter in ray['data']:
                                print('                {} = {}'.format(letter, ray['data'][letter][0:10]))
                        print('>>')
                    if ray['sweepEnd']:
                        # Call the collection of algorithms
                        for obj in self.algorithmObjects:
                            obj.process(self.sweep)
                            print('------')
                        print('\n')
                    # Gather all products
                    for letter in ['Z', 'V', 'W', 'D', 'P', 'R']:
                        if letter in self.sweep.products and letter in ray['data']:
                            self.sweep.products[letter][ii, 0:ng] = ray['data'][letter][0:ng]

        self.socket.close()

    def close(self):
        self.socket.close()
        
    def __del__(self):
        self.close()
