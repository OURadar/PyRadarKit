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
    MOMENT_DATA = 109
    ALERT_MESSAGE = 110
    CONFIG = 111

# Each delimiter has 16-bit type, 16-bit subtype, 32-bit raw size, 32-bit decoded size and 32-bit padding
RKNetDelimiter = b'HHIII'

# Generic functions
def test(payload, debug=False):
    return rkstruct.test(payload, debug=debug)

def init():
    rkstruct.init()

def showColors():
    rkstruct.showColors()

# A sweep encapsulation
class Sweep(object):
    """
        An object that encapsulate a sweep
    """
    def __init__(self):
        self.azimuth = 0.0
        self.elevation = 0.0
        self.sweepType = "PPI"
        self.rayCount = 0
        self.gateCount = 0
        self.products = {
            # 'Zi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            # 'Vi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            # 'Wi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            # 'Di': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            # 'Pi': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
            # 'Ri': N.zeros((360, CONSTANTS.MAX_GATES), dtype=N.uint8),
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

        rkstruct.init()
        
        # Initialize an empty list of algorithms
        self.algorithms = []
        self.sweep = Sweep()

        print('\033[38;5;226;48;5;24m              \033[0m')
        print('\033[38;5;226;48;5;24m  PyRadarKit  \033[0m')
        print('\033[38;5;226;48;5;24m              \033[0m')

    """
        Receives a frame: a network delimiter and the following payload described by the delimiter
        This method always finishes the frame reading
    """
    def _recv(self):
        try:
            length = self.socket.recv_into(self.netDelimiter, CONSTANTS.PACKET_DELIM_SIZE)
            if length != CONSTANTS.PACKET_DELIM_SIZE:
                raise ValueError('Length should be {}, not {}'.format(CONSTANTS.PACKET_DELIM_SIZE, length))
            delimiter = struct.unpack(RKNetDelimiter, self.netDelimiter)

            # 1st component: 16-bit type
            # 2nd component: 16-bit subtype (not used)
            # 3rd component: 32-bit size
            # 4th component: 32-bit decoded size (not used)
            payloadType = delimiter[0]
            payloadSize = delimiter[2]
            self.latestPayloadType = payloadType

            if payloadType != NETWORK_PACKET_TYPE.BEACON and payloadType != NETWORK_PACKET_TYPE.MOMENT_DATA:
                print('Delimiter type {} of size {}'.format(payloadType, payloadSize))

            # Beacon is 0 size, data payload otherwise
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

    """
        Start the server
    """
    def start(self):
        self.active = True

        # Loop through all the files under 'algorithms' folder
        if self.verbose:
            print('Loading algorithms ...')
        self.algorithmObjects = []
        for script in glob.glob('algorithms/*.py'):
            basename = os.path.basename(script)[:-3]
            mod = __import__(basename)
            obj = getattr(mod, 'main')()
            self.algorithmObjects.append(obj)
            if self.verbose:
                print(' • \033[38;5;220m{0:16s}\033[0m -> {1}'.format(basename, obj.name))

        # Connect to the host
        self.reconnect()

    """
        Stop the server
    """
    def stop(self):
        self.active = False

    """
        Make a network connection
    """
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

            # Request status, Z, V, W, D, P and R
            self.socket.send(b'sZVWDPR\r\n')

            # Keep reading while active
            while self.active:
                self._recv()
                if self.latestPayloadType == NETWORK_PACKET_TYPE.MOMENT_DATA:
                    # Parse the ray
                    ray = rkstruct.parse(self.payload, verbose=self.verbose)
                    # Gather the ray into a sweep
                    ii = int(ray['azimuth'])
                    ng = min(ray['gateCount'], CONSTANTS.MAX_GATES)
                    if self.verbose > 1:
                        print('   \033[38;5;226;48;5;24m PyRadarKit \033[0m \033[38;5;226mEL {0:0.2f} deg   AZ {1:0.2f} deg\033[0m -> {2} / {3}'.format(ray['elevation'], ray['azimuth'], ii, ray['sweepEnd']))
                        N.set_printoptions(formatter={'float': '{: 5.1f}'.format})
                        for letter in self.sweep.products.keys():
                            if letter in ray['data']:
                                print('                {} = {}'.format(letter, ray['data'][letter][0:10]))
                        print('>>')
                    if ray['sweepEnd']:
                        # Use this end ray only if it is not a begin ray and accumulated count < 360
                        if ray['sweepBegin'] == False and k < 360:
                            # Gather all products
                            for letter in self.sweep.products.keys():
                                if letter in ray['data']:
                                    self.sweep.products[letter][ii, 0:ng] = ray['data'][letter][0:ng]
                        # Call the collection of algorithms
                        for obj in self.algorithmObjects:
                            obj.process(self.sweep)
                        print('')
                    # Zero out all data when a sweep begin is encountered
                    if ray['sweepBegin']:
                        self.sweep.rayCount = 0
                        self.sweep.gateCount = ng
                        for letter in self.sweep.products.keys():
                            self.sweep.products[letter][:] = 0
                    # Gather all products
                    for letter in self.sweep.products.keys():
                        if letter in ray['data']:
                            self.sweep.products[letter][ii, 0:ng] = ray['data'][letter][0:ng]
                    self.sweep.rayCount += 1

        self.socket.close()

    """
        Close the socket
    """
    def close(self):
        self.socket.close()
        
    """
        Deallocate
    """
    def __del__(self):
        self.close()
