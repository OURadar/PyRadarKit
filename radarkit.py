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

__all__ = ['IP_ADDRESS', 'RADAR_PORT']

logger = logging.getLogger(__name__)

IP_ADDRESS = '127.0.0.1'
RADAR_PORT = 10000
BUFFER_SIZE = 10240
PACKET_DELIM_SIZE = 16
MAX_GATES = 4096

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

netType = b'H'
subType = b'H'
packedSize = b'I'
decodedSize = b'I'
delimiterPad = b'HH'
RKNetDelimiter = netType + subType + packedSize + decodedSize + delimiterPad
del netType, subType, packedSize, decodedSize, delimiterPad

# Generic functions
def test(payload, debug=False):
    print('debug = {}'.format(debug))
    return rkstruct.test(payload, debug=debug)

def init():
    rkstruct.init()

def showColors():
    rkstruct.showColors()

class Sweep(object):
    """An object that encapsulate a sweep
    """
    def __init__(self):
        self.azimuth = 0.0
        self.elevation = 0.0
        self.sweepType = "PPI"
        self.products = {
            'Z': N.zeros((360, MAX_GATES), dtype=N.float),
            'V': N.zeros((360, MAX_GATES), dtype=N.float)
        }

# Radar class
class Radar(object):
    """Handles the connection to the radar (created by RadarKit)
    This class allows to retrieval of base data from the radar
    """
    def __init__(self, ipAddress=IP_ADDRESS, port=RADAR_PORT, timeout=2, verbose=0):
        self.ipAddress = ipAddress
        self.port = port
        self.timeout = timeout
        self.netDelimiter = bytearray(PACKET_DELIM_SIZE)
        self.payload = bytearray(BUFFER_SIZE)
        self.latestPayloadType = 0

        self.verbose = verbose
        print('verbose = {}'.format(self.verbose))

        # Initialize an empty list of algorithms
        self.algorithms = []
        self.sweep = Sweep()

    def _recv(self):
        try:
            length = self.socket.recv_into(self.netDelimiter, PACKET_DELIM_SIZE)
            if length != PACKET_DELIM_SIZE:
                raise ValueError('Length should be {}, not {}'.format(PACKET_DELIM_SIZE, length))
            delimiter = struct.unpack(RKNetDelimiter, self.netDelimiter)

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
            print('File {} -> {} -> {}'.format(script, basename, obj.name()))

        self.reconnect()

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

            self.socket.send(b'sz\r\n')

            while self.active:
                self._recv()
                if self.latestPayloadType == NETWORK_PACKET_TYPE.RAY_DISPLAY:
                    # Parse the ray
                    ray = rkstruct.parse(self.payload, verbose=self.verbose)
                    # Gather the ray into a sweep
                    ii = int(ray['azimuth'])
                    ng = min(ray['gateCount'], MAX_GATES)
                    if ray['sweepEnd']:
                        for obj in self.algorithmObjects:
                            obj.process(self.sweep)
                            print('--------')
                        print('\n')
                    #self.sweep[ii, 0:ng] = ray['data'][0:ng]
                    self.sweep.products['Z'][ii, 0:ng] = ray['data'][0:ng]
                    if self.verbose > 1:
                        print('========')
                        print('    PyRadarKit: EL {0:0.2f} deg   AZ {1:0.2f} deg -> {2}'.format(ray['elevation'], ray['azimuth'], ii), end='')
                        print('   Zi = {} / {}'.format(ray['data'][0:10:], ray['sweepBegin']))
                    # Call the collection of processes

    def close(self):
        self.socket.close()
        
    def __del__(self):
        self.close()
