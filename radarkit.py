"""
    Python wrapper for C functions to interact with RadarKit
"""

import os
import re
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
import json

import rkstruct as rk

# Some global variables / functions
logger = logging.getLogger(__name__)
version_info = rk.version()

# Each delimiter has 16-bit type, 16-bit subtype, 32-bit raw size, 32-bit decoded size and 32-bit padding
RKNetDelimiterFormat = b'HHIII'

# Constants
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
    USER_PRODUCT_DESCRIPTION = 116

class COLOR:
    reset = "\033[0m"
    red = "\033[38;5;196m"
    orange = "\033[38;5;214m"
    yellow = "\033[38;5;226m"
    lime = "\033[38;5;118m"
    green = "\033[38;5;46m"
    teal = "\033[38;5;49m"
    iceblue = "\033[38;5;51m"
    skyblue = "\033[38;5;39m"
    blue = "\033[38;5;27m"
    purple = "\033[38;5;99m"
    indigo = "\033[38;5;201m"
    hotpink = "\033[38;5;199m"
    pink = "\033[38;5;213m"
    salmon = "\033[38;5;210m"
    python = "\033[38;5;226;48;5;24m"
    radarkit = "\033[38;5;15;48;5;124m"

# Generic functions
def test(payload, debug=False):
    return rk.test(payload, debug=debug)

def showColors():
    rk.showColors()

def read(filename, verbose=0):
    """
        Read a sweep from a netcdf file
    """
    return rk.read(filename, verbose=0)

def showArray(d, letter='U'):
    formatDesc = ' {:6.2f}'
    j = 0
    print('    {} = [ {} ... {} ]'.format(colorize(letter, COLOR.yellow),
                                              ' '.join([formatDesc.format(x) for x in d[j, :3]]),
                                              ' '.join([formatDesc.format(x) for x in d[j, -3:]])))
    for j in range(1, 3):
        print('        [ {} ... {} ]'.format(' '.join([formatDesc.format(x) for x in d[j, :3]]),
                                             ' '.join([formatDesc.format(x) for x in d[j, -3:]])))
    print('        [  ...')
    for j in range(-3, 0):
        print('        [ {} ... {} ]'.format(' '.join([formatDesc.format(x) for x in d[j, :3]]),
                                             ' '.join([formatDesc.format(x) for x in d[j, -3:]])))

def colorize(string, color):
    return '{}{}{}'.format(color, string, COLOR.reset)

# An algorithm encapsulation
class Algorithm(object):
    def __init__(self):
        self.name = 'Algorithm'
        self.symbol = 'U'
        self.active = False
        self.b = 1.0
        self.w = 0.0
        self.minValue = 0.0
        self.maxValue = 100.0
        self.productId = 0
        self.verbose = 0

    def __str__(self):
        return '{} -> {}   {} = {}'.format(colorize(self.name, COLOR.salmon),
                                        colorize(self.symbol, COLOR.yellow),
                                        colorize('active', COLOR.orange), colorize(self.active, COLOR.purple)
                                        )

    def description(self):
        dic = {'name': self.name, 'symbol': self.symbol, 'b': self.b, 'w': self.w}
        return json.dumps(dic)

    # Every algorithm should have this function defined
    def process(self, sweep):
        logger.info('Algorithm {}   {} = {}   {} = {}'.format(self,
                                                    colorize('rays', COLOR.orange), colorize(sweep.rayCount, COLOR.lime),
                                                    colorize('gates', COLOR.orange), colorize(sweep.gateCount, COLOR.lime)
                                                    ))

# A sweep encapsulation
class Sweep(object):
    """
        An object that encapsulate a sweep
    """
    def __init__(self, rays=360, gates=CONSTANTS.MAX_GATES):
        self.name = 'RadarKit'
        self.configId = 0
        self.rayCount = 0
        self.gateCount = 0
        self.sweepAzimuth = 0.0
        self.sweepElevation = 0.0
        self.gateSizeMeters = 1.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0
        self.sweepType = 'PPI'
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
        self.latestPayloadSize = 0
        self.registerString = ''

        rk.init()

        self._showName()
        
        # Initialize an empty list of algorithms
        self.algorithms = []
        self.sweep = Sweep()

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

        logger.info('Started.')

    def _showName(self):
        print('Version {}'.format(sys.version_info))
        # Size of the current terminal
        rows, columns = os.popen('stty size', 'r').read().split()
        print(colorize('\n\n{}\n'.format('RadarKit'.center(int(columns), ' ')), COLOR.radarkit))
        print(colorize('\n\n{}\n'.format('PyRadarKit'.center(int(columns), ' ')), COLOR.python) + '\n')

    """
        Receives a frame: a network delimiter and the following payload described by the delimiter
        This method always finishes the frame reading
    """
    def _recv(self):
        try:
            k = 0;
            length = 0
            toRead = CONSTANTS.PACKET_DELIM_SIZE
            anchor = memoryview(self.netDelimiterBytes)
            while toRead and k < 100:
                try:
                    length = self.socket.recv_into(anchor, toRead)
                except ConnectionResetError:
                    logger.exception('Connection reset during network delimiter.')
                    break
                except (ValueError) as e:
                    logger.exception(e)
                    break

                anchor = anchor[length:]
                toRead -= length
                k += 1
            if toRead:
                logger.warning('Length should be {}, not {}   k = {}'.format(CONSTANTS.PACKET_DELIM_SIZE, length, k))
                return False
            delimiter = self.netDelimiterStruct.unpack(self.netDelimiterBytes)

            # 1st component: 16-bit type
            # 2nd component: 16-bit subtype (not used)
            # 3rd component: 32-bit size
            # 4th component: 32-bit decoded size (not used)
            self.latestPayloadType = delimiter[0]
            self.latestPayloadSize = delimiter[2]

            if self.verbose > 2:
                logger.info('Delimiter: type {} size {}   k = {}'.format(self.latestPayloadType, self.latestPayloadSize, k))

            # Beacon is 0 size, data payload otherwise
            if self.latestPayloadSize > 0:
                k = 0
                toRead = self.latestPayloadSize
                anchor = memoryview(self.payload)
                while toRead and k < 100:
                    try:
                        length = self.socket.recv_into(anchor, toRead)
                    except ConnectionResetError:
                        logger.info('Connection reset during payload.')
                        break
                    except (ValueError) as e:
                        logger.exception(e)
                        break

                    anchor = anchor[length:]
                    toRead -= length
                    k += 1
            else:
                if not self.connected:
                    self.connected = True
                    logger.info('Connected.')
            return True

        except (socket.timeout, ValueError) as e:
            logger.exception(e)
            return False

    """
        Interpret the network payload
    """
    def _interpretPayload(self):
        if self.latestPayloadType == NETWORK_PACKET_TYPE.MOMENT_DATA:
            # Parse the ray
            ray = rk.parseRay(self.payload, verbose=self.verbose)
            # Gather the ray into a sweep
            ii = int(ray['azimuth'])
            ng = min(ray['gateCount'], CONSTANTS.MAX_GATES)
            if self.verbose > 1:
                print('   {} {} -> {2} / {3}'.format(colorize(' PyRadarKit ', COLOR.python),
                                                     colorize('EL {:0.2f} deg   AZ {:0.2f} deg'.format(ray['elevation'], ray['azimuth']), COLOR.yellow),
                                                     ii, ray['sweepEnd']))
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
                    # Call the collection of algorithms (deprecating... updated 6/13/2018: how?)
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
            sweepHeader = rk.parseSweepHeader(self.payload, verbose=self.verbose)
            #print(sweepHeader)
            self.sweep.name = sweepHeader['name']
            self.sweep.configId = sweepHeader['configId']
            self.sweep.rayCount = sweepHeader['rayCount']
            self.sweep.gateCount = sweepHeader['gateCount']
            self.sweep.gateSizeMeters = sweepHeader['gateSizeMeters']
            self.sweep.sweepAzimuth = sweepHeader['sweepAzimuth']
            self.sweep.sweepElevation = sweepHeader['sweepAzimuth']
            self.sweep.latitude = sweepHeader['latitude']
            self.sweep.longitude = sweepHeader['longitude']
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
            # Show some sweep info
            logger.info('New sweep   {} = {}   {} = {}   {} = {}   moments = {}'.format(colorize('configId', COLOR.orange), colorize(self.sweep.configId, COLOR.lime),
                                                                                        colorize('rays', COLOR.orange), colorize(self.sweep.rayCount, COLOR.lime),
                                                                                        colorize('gates', COLOR.orange), colorize(self.sweep.gateCount, COLOR.lime),
                                                                                        self.sweep.validSymbols))

        elif self.latestPayloadType == NETWORK_PACKET_TYPE.SWEEP_RAY:

            # Parse the individual ray of a sweep
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
                for symbol, obj in self.algorithmObjects.items():
                    userProduct = obj.process(self.sweep)
                    if obj.active is True:
                        if userProduct is None:
                            logger.exception('Expected a product from {}', obj)
                            continue
                        if self.verbose > 1:
                            logger.info('Sending product ...\n')
                        # 1st component: 16-bit type
                        # 2nd component: 16-bit subtype (not used)
                        # 3rd component: 32-bit size
                        # 4th component: 32-bit decoded size (not used)
                        # 5th component: 32-bit padding
                        bytes = self.sweep.gateCount * self.sweep.rayCount * 4
                        values = (NETWORK_PACKET_TYPE.USER_SWEEP_DATA, 0, bytes, bytes, 0)
                        packet = self.netDelimiterStruct.pack(*values)
                        r = self.socket.sendall(packet, CONSTANTS.PACKET_DELIM_SIZE);
                        if r is not None:
                            logger.exception('Error sending netDelimiter.')
                        # Product header
                        # sweepId ?

                        # Data array in plain float
                        r = self.socket.sendall(userProduct.astype('f').tobytes())
                        if r is not None:
                            logger.exception('Error sending userProduct.')
                        logger.info('{}   User product {}{}{} sent'.format(obj.name, COLOR.yellow, obj.symbol, COLOR.reset))

        elif self.latestPayloadType == NETWORK_PACKET_TYPE.COMMAND_RESPONSE:

            responseString = self.payload[0:self.latestPayloadSize].decode('utf-8').rstrip('\r\n\x00')
            logger.info('Command -> {}'.format(colorize(responseString, COLOR.skyblue)))
            curlyBracketPosition = responseString.find('{')
            if curlyBracketPosition > 0:
                payloadDict = json.loads(responseString[curlyBracketPosition:])
                if payloadDict['type'] == 'productDescription':
                    symbol = payloadDict['symbol']
                    if symbol is not None:
                        self.algorithmObjects[symbol].productId = payloadDict['pid']
                        logger.info('Product {} registered   {} = {}'.format(
                            colorize(symbol, COLOR.yellow),
                            colorize('productId', COLOR.orange),
                            colorize(payloadDict['pid'], COLOR.purple)))

    """
        Start the server
    """
    def start(self):
        # Loop through all the files under 'algorithms' folder
        logger.info('Loading algorithms ...')
        w = 1
        self.algorithmObjects = {}
        for script in glob.glob('algorithms/*.py'):
            basename = os.path.basename(script)
            mod = __import__(basename[:-3])
            obj = getattr(mod, 'main')()
            obj.basename = basename
            self.algorithmObjects.update({obj.symbol: obj})
            w = max(w, len(obj.basename))
            if (obj.active):
                self.registerString += 'u {};'.format(obj.description())
        # Remove the last ';'
        self.registerString = self.registerString[:-1]
        # Build a format so that the basename uses the widest name width
        stringFormat = '> {} - {}{:' + str(w) + 's}{} -> {}'
        for symbol, obj in self.algorithmObjects.items():
            logger.info(stringFormat.format(colorize(obj.symbol, COLOR.yellow),
                                            COLOR.lime, obj.basename, COLOR.reset, obj.name))
        # Composite registration string is built at this point
        logger.info('Registration = {}'.format(colorize(self.registerString, COLOR.salmon)))
        # Prepend data stream request
        greetCommand = 'sYU;' + self.registerString + '\r\n'
        logger.info('First packet - {}'.format(colorize(greetCommand.encode('utf-8'), COLOR.salmon)))

        # Connect to the host and reconnect until it has been set not active
        self.active = True
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

            # Send the greeting packet
            self.socket.sendall(greetCommand.encode())

            # Keep reading while active
            while self.active:
                if self._recv() == True:
                    self._interpretPayload()
                else:
                    break;

        logger.info('Connection from {} terminated.'.format(self.ipAddress))
        self.socket.close()

    """
        Stop the server
    """
    def stop(self):
        self.active = False

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
