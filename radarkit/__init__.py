"""
    Python wrapper for C functions to interact with RadarKit
"""

# Standard libraries
import os
import re
import sys
import enum
import glob
import math
import time
import logging
import threading
import datetime
import logging
import socket
import struct
import json

# Additional libraries
import numpy as N
import scipy as S

# From the PyRadarKit framework
from .rk import *
from .misc import *
from .test import *

# Some global objects / variables / functions
logger = logging.getLogger(__name__)
version_info = '1.0'

# All algorithms are located under the folder 'algorithms'
sys.path.insert(0, 'algorithms')

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

# An algorithm encapsulation
class Algorithm(object):
    def __init__(self, verbose=0):
        self.name = 'Algorithm'
        self.symbol = 'X'
        self.unit = ''
        self.colormap = 'Default'
        self.active = False
        self.b = 1.0
        self.w = 0.0
        self.minValue = None
        self.maxValue = None
        self.productId = 0
        self.verbose = verbose

    def __str__(self):
        return '{} -> {}   {}'.format(colorize(self.name, COLOR.salmon),
                                      colorize(self.symbol, COLOR.yellow),
                                      variableInString('active', self.active))

    def description(self):
        dic = {'name': self.name, 'symbol': self.symbol, 'unit': self.unit, 'colormap': self.colormap, 'b': self.b, 'w': self.w}
        return json.dumps(dic)

    # Every algorithm should have this function defined
    def process(self, sweep):
        if self.verbose > 1:
            logger.info('Algorithm {}   {}   {}'.format(self,
                                                        variableInString('rays', sweep.rayCount),
                                                        variableInString('gates', sweep.gateCount)))
        else:
            logger.info('Algorithm {}'.format(self))

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
    def __init__(self, ipAddress=CONSTANTS.IP, port=CONSTANTS.PORT, timeout=2, streams=None, algorithmFolder='algorithms', verbose=0):
        self.ipAddress = ipAddress
        self.port = port
        self.timeout = timeout
        self.streams = streams
        self.verbose = verbose
        self.active = False
        self.wantActive = False
        self.algorithmFolder = algorithmFolder[:-1] if algorithmFolder.endswith('/') else algorithmFolder
        print('{}'.format(variableInString('algorithmFolder', self.algorithmFolder)))
        self.netDelimiterBytes = bytearray(CONSTANTS.PACKET_DELIM_SIZE)
        # Each netlimiter has:
        # 1st component: 16-bit type
        # 2nd component: 16-bit subtype (not used)
        # 3rd component: 32-bit size
        # 4th component: 32-bit decoded size (not used)
        # 5th component: 32-bit padding
        self.netDelimiterStruct = struct.Struct(b'HHIII')
        self.netDelimiterValues = [0, 0, 0, 0, 0]
        self.payload = bytearray(CONSTANTS.BUFFER_SIZE)
        self.latestPayloadType = 0
        self.latestPayloadSize = 0
        self.registerString = ''

        # Initialize the C extension
        init()

        # Initialize a bunch to things
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
            # 5th component: 32-bit padding
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
            ray = parseRay(self.payload, verbose=self.verbose)
            # Gather the ray into a sweep
            ii = int(ray['azimuth'])
            ng = min(ray['gateCount'], CONSTANTS.MAX_GATES)
            if self.verbose > 2:
                print('   {} {} -> {} / sweepEnd = {}'.format(colorize(' PyRadarKit ', COLOR.python),
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
            sweepHeader = parseSweepHeader(self.payload, verbose=self.verbose)
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
            for symbol in self.sweep.validSymbols:
                self.sweep.products[symbol] = N.zeros((self.sweep.rayCount, self.sweep.gateCount), dtype=N.float)
            # Show some sweep info
            logger.info('New sweep   {}   {}   {}   {}'.format(variableInString('configId', self.sweep.configId),
                                                               variableInString('rays', self.sweep.rayCount),
                                                               variableInString('gates', self.sweep.gateCount),
                                                               variableInString('symbols', self.sweep.validSymbols)))

        elif self.latestPayloadType == NETWORK_PACKET_TYPE.SWEEP_RAY:

            # Parse the individual ray of a sweep
            ray = parseRay(self.payload, verbose=self.verbose)
            k = self.sweep.receivedRayCount;
            self.sweep.azimuth[k] = ray['azimuth']
            self.sweep.elevation[k] = ray['elevation']
            for symbol in self.sweep.validSymbols:
                self.sweep.products[symbol][k, 0:self.sweep.gateCount] = ray['moments'][symbol][0:self.sweep.gateCount]
            if self.verbose > 2:
                print('   {} {} -> {} / {}'.format(colorize(' PyRadarKit ', COLOR.python),
                                                   colorize('EL {:0.2f} deg   AZ {:0.2f} deg'.format(self.sweep.elevation[k], self.sweep.azimuth[k]), COLOR.yellow),
                                                   k, self.sweep.rayCount))
                N.set_printoptions(formatter={'float': '{: 5.1f}'.format})
                for symbol in self.sweep.products.keys():
                    print('                {} = {}'.format(symbol, self.sweep.products[symbol][k, 0:10]))
                print('>>')
            self.sweep.receivedRayCount += 1
            if self.sweep.receivedRayCount == self.sweep.rayCount:
                # Call the collection of algorithms
                for symbol, obj in self.algorithmObjects.items():
                    userProductData = obj.process(self.sweep)
                    userProductDesc = json.dumps({
                                                 'productId': obj.productId,
                                                 'configId': self.sweep.configId
                                                 }).encode('utf-8')
                    if obj.active:
                        if userProductData is None:
                            logger.exception('Expected a product from {}'.format(obj))
                            continue
                        if self.verbose > 1:
                            logger.info('Sending product ...')
                        # Network delimiter (see above)
                        bytes = len(userProductDesc)
                        values = (NETWORK_PACKET_TYPE.USER_PRODUCT_DESCRIPTION, 0, bytes, bytes, 0)
                        delimiterForUserProductDesc = self.netDelimiterStruct.pack(*values)
                        # Network delimiter (see above)
                        bytes = self.sweep.gateCount * self.sweep.rayCount * 4
                        values = (NETWORK_PACKET_TYPE.USER_SWEEP_DATA, 0, bytes, bytes, 0)
                        delimiterForData = self.netDelimiterStruct.pack(*values)
                        # Data array in plain float array
                        r = self.socket.sendall(delimiterForUserProductDesc + userProductDesc + delimiterForData + userProductData.astype('f').tobytes())
                        if r is not None:
                            logger.exception('Error sending userProduct.')
                        logger.info('User product {} sent'.format(colorize(obj.symbol, COLOR.yellow)))

        elif self.latestPayloadType == NETWORK_PACKET_TYPE.COMMAND_RESPONSE:

            # Command response is a string
            responseString = self.payload[0:self.latestPayloadSize].decode('utf-8').rstrip('\r\n\x00')
            logger.info('Response = {}'.format(colorize(responseString, COLOR.skyblue)))
            curlyBracketPosition = responseString.find('{')
            if curlyBracketPosition > 0:
                payloadDict = json.loads(responseString[curlyBracketPosition:])
                if payloadDict['type'] == 'productDescription':
                    symbol = payloadDict['symbol']
                    if symbol is not None:
                        self.algorithmObjects[symbol].productId = payloadDict['pid']
                        logger.info('Product {} registered   {}'.format(colorize(symbol, COLOR.yellow),
                                                                        variableInString('productId', payloadDict['pid'])))

    """
        The run loop
    """
    def _runLoop(self):
        if not self.streams:
            # Prepend data stream request
            #greetCommand = 'sYUXCOQ;' + self.registerString + '\r\n'
            greetCommand = 'sYUCO;' + self.registerString + '\r\n'
        else:
            greetCommand = 's' + self.streams + '\r\n'

        greetCommand = greetCommand.encode('utf-8')
        logger.debug('First packet = {}'.format(colorize(greetCommand, COLOR.salmon)))
        # Connect to the host and reconnect until it has been set not to wantActive
        try:
            while self.wantActive:
                self.active = False
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
                self.socket.sendall(greetCommand)
                # Keep reading while wantActive
                while self.wantActive:
                    if self._recv() == True:
                        self._interpretPayload()
                    else:
                        logger.info('Server disconnected.')
                        t = 30
                        while t > 0:
                            if self.verbose > 1 and t % 10 == 0:
                                print('Retry in {0:.0f} seconds ...\r'.format(t * 0.1))
                            time.sleep(0.1)
                            t -= 1
                        self.socket.close()
                        break;
                    if not self.active:
                        self.active = True
        except:
            print('Outside runloop')
        # Outside of the busy loop
        logger.info('Connection from {} terminated.'.format(self.ipAddress))
        self.socket.close()
        self.active = False

    """
        Start the server
    """
    def start(self):
        # Loop through all the files under 'algorithms' folder
        logger.info('Loading algorithms ...')
        w = 1
        self.algorithmObjects = {}
        for script in glob.glob('{}/*.py'.format(self.algorithmFolder)):
            basename = os.path.basename(script)
            mod = __import__(basename[:-3])
            obj = getattr(mod, 'main')(verbose=self.verbose)
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
        self.wantActive = True
        threading.Thread(target=self._runLoop).start()

    """
        Stop the server
    """
    def stop(self):
        logger.info('Deactivating radar ...')
        self.wantActive = False
        k = 0
        while self.active and k < 20:
            time.sleep(0.1)
            k += 1
        if k >= 20:
            logger.info('Force exit.')
        logger.info('Done.')

    """
        Wait while the radar is active
    """
    def wait(self):
        while (self.active):
            time.sleep(0.1)
    
    """
        Close the socket
    """
    def close(self):
        wantActive = False
        if (self.socket):
            self.socket.close()
        
    """
        Deallocate
    """
    def __del__(self):
        self.close()
