"""
    Python wrapper for C functions to interact with RadarKit
"""

import os
import sys
import enum
import glob
import math
import time
import datetime
import logging
import socket
import struct
import numpy as N
import scipy as S

import rkstruct as rk

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
    SWEEP_HEADER = 113
    SWEEP_RAY = 114
    USER_SWEEP_DATA = 115

# Each delimiter has 16-bit type, 16-bit subtype, 32-bit raw size, 32-bit decoded size and 32-bit padding
RKNetDelimiterFormat = b'HHIII'

# Generic functions
def test(payload, debug=False):
    return rk.test(payload, debug=debug)

def init():
    rk.init()

def showColors():
    rk.showColors()

def read(filename, verbose=0):
    return rk.read(filename, verbose=0)

# An algorithm encapsulation
class Algorithm(object):
    def __init__(self):
        self.name = 'Algorithm'
        self.active = False

    def __str__(self):
        return '{}   active {}'.format(self.name, self.active)

    # Every algorithm should have this function defined
    def process(self, sweep):
        print('{}:  R = {}  G = {}'.format(self.name, sweep.rayCount, sweep.gateCount))

# A sweep encapsulation
class Sweep(object):
    """
        An object that encapsulate a sweep
    """
    def __init__(self, rays=360, gates=CONSTANTS.MAX_GATES):
        self.azimuth = 0.0
        self.elevation = 0.0
        self.sweepType = "PPI"
        self.rayCount = 0
        self.gateCount = 0
        self.receivedRayCount = 0
        self.validSymbols = []
        self.products = {
            'Z': N.zeros((rays, gates), dtype=N.float),
            'V': N.zeros((rays, gates), dtype=N.float),
            'W': N.zeros((rays, gates), dtype=N.float),
            'D': N.zeros((rays, gates), dtype=N.float),
            'P': N.zeros((rays, gates), dtype=N.float),
            'R': N.zeros((rays, gates), dtype=N.float)
        }

    """
        Read a sweep from a netcdf file
    """
    # def read(self, filename):
        # file = netcdf.Dataset(filename, 'r')
        # print file.data_model
        # self.globalAttributes = file.ncattrs()
        # dims = [dim for dim in file.dimensions]
        # print "NetCDF dimension information:"
        # for dim in dims:
        #     print "\tName:", dim 
        #     print "\t\tsize:", len(file.dimensions[dim])
        #     print_ncattr(dim)
        # vars = [var for var in file.variables]
        # for var in vars:
        #     if var not in dims:
        #         print '\tName:', var
        #         print "\t\tdimensions:", file.variables[var].dimensions
        #         print "\t\tsize:", file.variables[var].size
        #         #print_ncattr(var)
        # file.close()


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
        self.netDelimiterBytes = bytearray(CONSTANTS.PACKET_DELIM_SIZE)
        self.netDelimiterStruct = struct.Struct(RKNetDelimiterFormat)
        self.netDelimiterValues = [0, 0, 0, 0, 0]
        self.payload = bytearray(CONSTANTS.BUFFER_SIZE)
        self.latestPayloadType = 0

        rk.init()
        
        # Initialize an empty list of algorithms
        self.algorithms = []
        self.sweep = Sweep()

        print('\033[38;5;226;48;5;24m              \033[0m')
        print('\033[38;5;226;48;5;24m  PyRadarKit  \033[0m')
        print('\033[38;5;226;48;5;24m              \033[0m')

        logFolder = 'log'
        if not os.path.exists(logFolder):
            os.makedirs(logFolder)

        logFile = logFolder + '/' + datetime.datetime.now().strftime('pyrk-%Y%m%d.log')
        logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%I:%M:%S')

        ch = logging.StreamHandler()
        if self.verbose > 0:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%I:%M:%S'))
        logger.addHandler(ch)

        logger.info('PyRadarKit started.')

    """
        Receives a frame: a network delimiter and the following payload described by the delimiter
        This method always finishes the frame reading
    """
    def _recv(self):
        try:
            k = 0;
            toRead = CONSTANTS.PACKET_DELIM_SIZE
            anchor = memoryview(self.netDelimiterBytes)
            while toRead and k < 100:
                length = self.socket.recv_into(anchor, toRead)
                anchor = anchor[length:]
                toRead -= length
                k += 1
            if toRead:
                raise ValueError('Length should be {}, not {}   k = {}'.format(CONSTANTS.PACKET_DELIM_SIZE, length, k))
            delimiter = self.netDelimiterStruct.unpack(self.netDelimiterBytes)

            # 1st component: 16-bit type
            # 2nd component: 16-bit subtype (not used)
            # 3rd component: 32-bit size
            # 4th component: 32-bit decoded size (not used)
            payloadType = delimiter[0]
            payloadSize = delimiter[2]
            self.latestPayloadType = payloadType

            if self.verbose > 2:
                print('Delimiter: type {} size {}   k = {}'.format(payloadType, payloadSize, k))

            # Beacon is 0 size, data payload otherwise
            if payloadSize > 0:
                k = 0
                toRead = payloadSize
                anchor = memoryview(self.payload)
                while toRead and k < 100:
                    length = self.socket.recv_into(anchor, toRead)
                    anchor = anchor[length:]
                    toRead -= length
                    k += 1
            else:
                if not self.connected:
                    self.connected = True
                    logger.info('Connected.')

        except (socket.timeout, ValueError) as e:
            logger.exception(e)
            raise OSError('Couldn\'t retrieve socket data')

    """
        Start the server
    """
    def start(self):
        # Loop through all the files under 'algorithms' folder
        logger.info('Loading algorithms ...')
        self.algorithmObjects = []
        for script in glob.glob('algorithms/*.py'):
            basename = os.path.basename(script)[:-3]
            mod = __import__(basename)
            obj = getattr(mod, 'main')()
            self.algorithmObjects.append(obj)
            logger.info(' - \033[38;5;220m{0:16s}\033[0m -> {1}'.format(basename, obj.name))

        # Connect to the host
        self.active = True
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
            self.connected = False
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            try:
                logger.info('Connecting {}:{}...'.format(self.ipAddress, self.port))
                self.socket.connect((self.ipAddress, self.port))
            except:
                t = 30
                while t > 0:
                    if self.verbose > 1 and t % 10 == 0:
                        print('Retry in {0:.0f} seconds ...\r'.format(t * 0.1))
                    time.sleep(0.1)
                    t -= 1
                self.socket.close()
                continue

            # Request status, Z, V, W, D, P and R
            #self.socket.send(b'sZVWDPR\r\n')
            self.socket.send(b'sYU\r\n')

            # Keep reading while active
            while self.active:
                self._recv()

                if self.latestPayloadType == NETWORK_PACKET_TYPE.MOMENT_DATA:
                    # Parse the ray
                    ray = rk.parseRay(self.payload, verbose=self.verbose)
                    # Gather the ray into a sweep
                    ii = int(ray['azimuth'])
                    ng = min(ray['gateCount'], CONSTANTS.MAX_GATES)
                    if self.verbose > 1:
                        print('   \033[38;5;226;48;5;24m PyRadarKit \033[0m \033[38;5;226mEL {0:0.2f} deg   AZ {1:0.2f} deg\033[0m -> {2} / {3}'.format(ray['elevation'], ray['azimuth'], ii, ray['sweepEnd']))
                        N.set_printoptions(formatter={'float': '{: 5.1f}'.format})
                        for letter in self.sweep.products.keys():
                            if letter in ray['moments']:
                                print('                {} = {}'.format(letter, ray['moments'][letter][0:10]))
                        print('>>')
                    if ray['sweepEnd']:
                        # Use this end ray only if it is not a begin ray and accumulated count < 360
                        if ray['sweepBegin'] == False and k < 360:
                            # Gather all products
                            for letter in self.sweep.products.keys():
                                if letter in ray['moments']:
                                    self.sweep.products[letter][ii, 0:ng] = ray['moments'][letter][0:ng]
                        # Call the collection of algorithms (deprecating)
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
                        if letter in ray['moments']:
                            self.sweep.products[letter][ii, 0:ng] = ray['moments'][letter][0:ng]
                    self.sweep.rayCount += 1

                elif self.latestPayloadType == NETWORK_PACKET_TYPE.SWEEP_HEADER:

                    # Parse the sweep
                    sweepHeader = rk.parseSweep(self.payload, verbose=self.verbose)
                    self.sweep.gateCount = sweepHeader['gateCount']
                    self.sweep.rayCount = sweepHeader['rayCount']
                    self.sweep.validSymbols = sweepHeader['moments']
                    self.sweep.range = N.zeros(self.sweep.gateCount, dtype=N.float)
                    self.sweep.azimuth = N.zeros(self.sweep.gateCount, dtype=N.float)
                    self.sweep.elevation = N.zeros(self.sweep.gateCount, dtype=N.float)
                    self.sweep.receivedRayCount = 0;
                    self.sweep.products = {}
                    if self.verbose > 1:
                        print('validSymbols = {}'.format(self.sweep.validSymbols))
                    for symbol in self.sweep.validSymbols:
                        self.sweep.products[symbol] = N.zeros((self.sweep.rayCount, self.sweep.gateCount), dtype=N.float)
                    print('  New sweep -> {} x {}   moments = {}'.format(self.sweep.rayCount, self.sweep.gateCount, sweepHeader['moments']))

                elif self.latestPayloadType == NETWORK_PACKET_TYPE.SWEEP_RAY:

                    ray = rk.parseRay(self.payload, verbose=self.verbose)
                    k = self.sweep.receivedRayCount;
                    self.sweep.azimuth[k] = ray['azimuth']
                    self.sweep.elevation[k] = ray['elevation']
                    for symbol in self.sweep.validSymbols:
                        self.sweep.products[symbol][k, 0:self.sweep.gateCount] = ray['moments'][symbol][0:self.sweep.gateCount]
                    if self.verbose > 1:
                        print('   \033[38;5;226;48;5;24m PyRadarKit \033[0m \033[38;5;226mEL {0:0.2f} deg   AZ {1:0.2f} deg\033[0m -> {2} / {3}'.format(self.sweep.elevation[k], self.sweep.azimuth[k], k, self.sweep.rayCount))
                        N.set_printoptions(formatter={'float': '{: 5.1f}'.format})
                        for symbol in self.sweep.products.keys():
                            print('                {} = {}'.format(symbol, self.sweep.products[symbol][k, 0:10]))
                        print('>>')
                    self.sweep.receivedRayCount += 1
                    if self.sweep.receivedRayCount == self.sweep.rayCount:
                        # Call the collection of algorithms
                        for obj in self.algorithmObjects:
                            userProduct = obj.process(self.sweep)
                            if obj.active is True:
                                print('    Sending product ...\n')
                                # 1st component: 16-bit type
                                # 2nd component: 16-bit subtype (not used)
                                # 3rd component: 32-bit size
                                # 4th component: 32-bit decoded size (not used)
                                # 5th component: 32-bit padding
                                bytes = self.sweep.gateCount * self.sweep.rayCount * 4
                                values = (NETWORK_PACKET_TYPE.USER_SWEEP_DATA, 0, bytes, bytes, 0)
                                packet = self.netDelimiterStruct.pack(*values)
                                self.socket.sendall(packet, CONSTANTS.PACKET_DELIM_SIZE);

#                                product should be a dictionary of:
#                                {'name', [Product Description],
#                                 'data', [Array, same size as Z]}
                                n = self.socket.sendall(userProduct.astype('f').tobytes())
                        print('')


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
