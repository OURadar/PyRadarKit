"""
    Python wrapper for C functions to interact with RadarKit
"""
#
#  PyRadarKit
#
#  Created by Boonleng Cheong
#

#
#  Updates
#
#  2.0.3  - 4/1/2019
#         - Attenuation correction algorithms: scwc, zphi
#         - Incorporated RadarKit 2.1.1
#         - Logging to be compatible with iRadar
#
#  2.0    - 3/7/2019
#         - Notebooks on algorithm development
#         - Notebooks on product routine development
#         - Product routine registration to RadarKit
#         - Confirmation with product Id from RadarKit
#
#  1.1    - 3/9/2018
#         - Added Chart class
#
#  1.0    - 11/29/2017
#         - Full cycle working
#         - Presentable main script
#
#  0.5    - 6/19/2017
#         - Some parts are working
#
#  0.1    - 6/3/2017
#         - Started
#

# Libraries
import re
import os
import sys
import glob
import math
import time
import array
import logging
import threading
import datetime
import logging
import socket
import struct
import json
import numpy as N
import scipy as S

# From the PyRadarKit framework
from .foundation import *
from .test import *
from .rk import *

#branch = os.popen('git rev-parse --abbrev-ref HEAD').read()
from ._version import __version__
if 'b' in version():
    __version__ += 'b'
    version_info = __version__

# Some global objects / variables / functions
logger = logging.getLogger('PyRadarKit')

# An algorithm encapsulation
class ProductRoutine(object):
    def __init__(self, verbose=0):
        self.key = 0
        self.name = 'Algorithm Name'
        self.productCount = 0
        self.productName = 'ExpIPi'
        self.productId = None
        self.symbol = '_'
        self.unit = 'unitless'
        self.cmap = 'Default'
        self.b = 1.0
        self.w = 0.0
        self.minValue = -999.9
        self.maxValue = +999.9
        self.verbose = verbose
        self.active = False

    def __str__(self):
        if self.productCount:
            infoString = '{} -> {}'.format(colorize(self.name, COLOR.iceblue),
                                           ', '.join([colorize(x, COLOR.yellow) for x in self.symbol]))
        else:
            infoString = colorize(self.name, COLOR.iceblue)
        if self.verbose > 1:
            infoString += '   ' + variableInString('active', self.active)
        return infoString

    def description(self):
        if self.productCount > 1:
            #print(self.productName)
            #print(self.symbol)
            #print(self.unit)
            #print(self.cmap)
            #print(self.w)
            #print(self.b)
            strings = []
            for name, symbol, unit, cmap, w, b in zip(self.productName, self.symbol, self.unit, self.cmap, self.w, self.b):
                dic = {'key': self.key, 'name': name, 'symbol': symbol, 'unit': unit, 'colormap': cmap, 'w': w, 'b': b}
                strings.append(json.dumps(dic).encode('utf-8'))
            return strings
        dic = {'key': self.key, 'name': self.productName, 'symbol': self.symbol, 'unit': self.unit, 'colormap': self.cmap, 'w': self.w, 'b': self.b}
        return json.dumps(dic)

    # Every algorithm should have this function defined
    def process(self, sweep):
        logger.info('Algorithm {}'.format(self))
        logger.debug('{}   {}'.format(variableInString('rays', sweep.rayCount), variableInString('gates', sweep.gateCount)))

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
            'Sh': N.zeros((rays, gates), dtype=N.float),
            'Z': N.zeros((rays, gates), dtype=N.float),
            'V': N.zeros((rays, gates), dtype=N.float),
            'W': N.zeros((rays, gates), dtype=N.float),
            'D': N.zeros((rays, gates), dtype=N.float),
            'P': N.zeros((rays, gates), dtype=N.float),
            'R': N.zeros((rays, gates), dtype=N.float),
            'Q': N.zeros((rays, gates), dtype=N.float)
        }

# Radar class
class Radar(object):
    """
        Handles the connection to the radar (created by RadarKit)
        This class allows to retrieval of base data from the radar
    """
    def __init__(self, ipAddress=CONSTANTS.IP, port=CONSTANTS.PORT, timeout=2, streams=None, productRoutinesFolder='productRoutines', verbose=0):
        self.ipAddress = ipAddress
        self.port = port
        self.timeout = timeout
        self.streams = streams
        self.verbose = verbose
        self.active = False
        self.wantActive = False
        self.productRoutinesFolder = productRoutinesFolder[:-1] if productRoutinesFolder.endswith('/') else productRoutinesFolder
        logger.debug('{}'.format(variableInString('productRoutinesFolder', self.productRoutinesFolder)))
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
        if self.verbose > 1:
            ch.setLevel(logging.INFO)
        elif self.verbose:
            ch.setLevel(logging.DEBUG)
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

            logger.debug('Delimiter: type {} size {}   k = {}'.format(self.latestPayloadType, self.latestPayloadSize, k))

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
                    for _, obj in self.algorithmObjects.items():
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
            stringOfSymbols = ', '.join([colorize(x, COLOR.yellow) for x in self.sweep.validSymbols])
            logger.debug('New sweep   {}   {}   {}   {}'.format(variableInString('configId', self.sweep.configId),
                                                                variableInString('rays', self.sweep.rayCount),
                                                                variableInString('gates', self.sweep.gateCount),
                                                                variableInString('symbols', stringOfSymbols)))

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
                N.set_printoptions(formatter={'float': '{:5.1f}'.format})
                for symbol in self.sweep.products.keys():
                    print('               {} = {}'.format(symbol.rjust(2, ' '), self.sweep.products[symbol][k, 0:10]))
                print('>>')
            self.sweep.receivedRayCount += 1
            # The rests are at the end of a sweep, early return if it is not end of sweep
            if not self.sweep.receivedRayCount == self.sweep.rayCount:
                return
            # Call the collection of algorithms
            processTime = 0.0
            deliveryTime = 0.0
            for key, obj in self.algorithmObjects.items():
                if obj.productCount:
                    stringOfProductIds = ', '.join([colorize(x, COLOR.lime) for x in obj.productId])
                    logger.debug('Calling algorithm key {} for {}   {}   {} = [{}] ...'.format(key, obj.symbol,
                                                                                               variableInString("productCount", obj.productCount),
                                                                                               colorize("productId", COLOR.orange), stringOfProductIds))
                else:
                    logger.debug('Calling algorithm key {}   {} ...'.format(key, variableInString("productCount", obj.productCount)))
                # Time the process
                tic = time.time()
                userProductData = obj.process(self.sweep)
                toc = time.time()
                processTime += (toc - tic)
                if not obj.active:
                    continue
                if userProductData is None or len(userProductData) == 0:
                    logger.exception('Expected product(s) from {}'.format(obj))
                    continue
                logger.debug('Sending products ...')
                if not isinstance(userProductData, list) and not isinstance(userProductData, tuple):
                    userProductData = [userProductData]
                userProductDesc = []
                for pid in obj.productId:
                    jsonString = json.dumps({'key': key, 'productId': pid, 'configId': self.sweep.configId}).encode('utf-8')
                    userProductDesc.append(jsonString)
                if not isinstance(obj.symbol, list) and not isinstance(obj.symbol, tuple):
                    symbols = [obj.symbol]
                else:
                    symbols = obj.symbol
                # Time the delivery
                tic = time.time()
                for data, desc, symbol in zip(userProductData, userProductDesc, symbols):
                    # Network delimiter (see above)
                    bytes = len(desc)
                    values = (NETWORK_PACKET_TYPE.USER_PRODUCT_DESCRIPTION, 0, bytes, bytes, 0)
                    delimiterForUserProductDesc = self.netDelimiterStruct.pack(*values)
                    # Network delimiter (see above)
                    bytes = self.sweep.gateCount * self.sweep.rayCount * 4
                    values = (NETWORK_PACKET_TYPE.USER_SWEEP_DATA, 0, bytes, bytes, 0)
                    delimiterForData = self.netDelimiterStruct.pack(*values)
                    # Data array in plain float array
                    r = self.socket.sendall(delimiterForUserProductDesc + desc + delimiterForData + data.astype('f').tobytes())
                    if r is not None:
                        logger.exception('Error sending userProduct.')
                    logger.debug('User product {} sent'.format(colorize(symbol, COLOR.yellow)))
                toc = time.time()
                deliveryTime += (toc - tic)
            # Log the total process and transmit time
            logger.info('Sweep @ {} completed   {} s   {} s'.format(variableInString('configId', self.sweep.configId),
                                                                    variableInString('processTime', processTime),
                                                                    variableInString('deliveryTime', deliveryTime)))

        elif self.latestPayloadType == NETWORK_PACKET_TYPE.COMMAND_RESPONSE:

            # Command response is a string
            responseString = self.payload[0:self.latestPayloadSize].decode('utf-8').rstrip('\r\n\x00')
            logger.debug('Response = {}'.format(colorize(responseString, COLOR.skyblue)))
            curlyBracketPosition = responseString.find('{')
            if curlyBracketPosition > 0:
                payloadDict = json.loads(responseString[curlyBracketPosition:])
                if payloadDict['type'] == 'productDescription':
                    key = payloadDict['key']
                    if key is not None:
                        #self.algorithmObjects[key].productId = payloadDict['pid']
                        pid = payloadDict['pid']
                        symbol = payloadDict['symbol']
                        idx = N.argmax([symbol == x for x in self.algorithmObjects[key].symbol])
                        logger.debug('Product {} registered   {}   {} ({})'.format(colorize(symbol, COLOR.yellow),
                                                                                  variableInString('key', key),
                                                                                  variableInString('productId', pid),
                                                                                  idx))
                        self.algorithmObjects[key].productId[idx] = pid

    """
        The run loop
    """
    def _runLoop(self):
        if self.streams is None:
            # Prepend data stream request
            greetCommand = 'sYUXCOQAH;' + self.registerString + '\r\n'
            #greetCommand = 'sYUCO;' + self.registerString + '\r\n'
        else:
            greetCommand = 's' + self.streams + '\r\n'

#        print('')
#        print(greetCommand)

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
                    logger.debug('Connecting {}:{}...'.format(self.ipAddress, self.port))
                    self.socket.connect((self.ipAddress, self.port))
                except:
                    t = 30
                    while t > 0 and self.wantActive:
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
                        logger.debug('Server disconnected.')
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
        except KeyboardInterrupt:
            print('Outside runloop KeyboardInterrupt')
        except:
            print('Outside runloop', sys.exc_info()[0])
        # Outside of the busy loop
        logger.debug('Connection from {} terminated.'.format(self.ipAddress))
        self.socket.close()
        self.active = False

    """
        Start the server
    """
    def start(self):
        # Loop through all the files under (productRoutines) folder
        logger.debug('Loading algorithms ...')
        sys.path.insert(0, self.productRoutinesFolder)
        w0 = 1
        w1 = 1
        uid = 1000
        self.algorithmObjects = {}
        for script in glob.glob('{}/*.py'.format(self.productRoutinesFolder)):
            basename = os.path.basename(script)
            #modduleName = '{}.{}'.format(self.productRoutinesFolder, basename[:-3])
            modduleName = basename[:-3]
            mod = __import__(modduleName)
            obj = getattr(mod, 'main')(verbose=self.verbose)
            obj.basename = basename
            obj.key = uid
            uid += 1
            obj.productId = [0 for _ in range(obj.productCount)]
            self.algorithmObjects.update({obj.key: obj})
            w0 = max(w0, len(obj.basename))
            w1 = max(w1, len(obj.name))
            if obj.productCount > 1:
                # Make sure the productNames, symbols, units, etc. are also lists
                if ((not isinstance(obj.productName, list) or not len(obj.productName) == obj.productCount) or
                    (not isinstance(obj.symbol, list) or not len(obj.symbol) == obj.productCount) or
                    (not isinstance(obj.unit, list) or not len(obj.unit) == obj.productCount) or
                    (not isinstance(obj.cmap, list) or not len(obj.cmap) == obj.productCount) or
                    (not isinstance(obj.w, list) or not len(obj.w) == obj.productCount) or
                    (not isinstance(obj.b, list) or not len(obj.b) == obj.productCount)):
                    logger.warning('Product routine should have productName, symbol, unit, cmap, w, b with same length.')
                if (obj.active):
                    for desc in obj.description():
                        self.registerString += 'u {};'.format(desc)
            else :
                if (obj.active):
                    self.registerString += 'u {};'.format(obj.description())
        # Remove the last ';'
        if len(self.registerString) > 8 and self.registerString[-1] is ';':
            self.registerString = self.registerString[:-1]
        # Build a format so that the basename uses the widest name width
        for key, obj in self.algorithmObjects.items():
            logger.info('> {}: {} - {} -> {}'.format(key,
                                                    colorize(obj.basename.ljust(w0, ' '), COLOR.lime),
                                                    colorize(obj.name.center(w1, ' '), COLOR.iceblue),
                                                    ', '.join([colorize(x, COLOR.yellow) for x in obj.symbol])))
        # Composite registration string is built at this point
        logger.debug('Registration = {}'.format(colorize(self.registerString, COLOR.salmon)))
        self.wantActive = True
        threading.Thread(target=self._runLoop).start()

    """
        Stop the server
    """
    def stop(self):
        logger.debug('Deactivating radar ...')
        self.wantActive = False
        k = 0
        while self.active and k < 20:
            time.sleep(0.1)
            k += 1
        if k >= 20:
            logger.debug('Forced exit.')
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
