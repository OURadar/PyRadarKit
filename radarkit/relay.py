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

# Some global objects / variables / functions
logger = logging.getLogger('iRadar')

class Relay(object):
    '''
        Relays an incoming connection to the radar (RadarKit)
    '''
    def __init__(self, relay=None, address=CONSTANTS.IP, port=CONSTANTS.PORT, timeout=2, verbose=0):
        self.address = address
        self.port = port
        self.relay = relay
        self.socket = None
        self.timeout = timeout
        self.verbose = verbose
        self.wantActive = False
        self.active = False
        self.payload = bytearray(CONSTANTS.BUFFER_SIZE)
        self.netDelimiterBytes = bytearray(CONSTANTS.PACKET_DELIM_SIZE)
        self.netDelimiterStruct = struct.Struct(b'HHIII')
        self.netDelimiterValues = [0, 0, 0, 0, 0]
        self.latestPayloadType = 0
        self.latestPayloadSize = 0
        self.registerString = ''

        # Initialize the C extension
        init()

        # Initialize a bunch to things
        logger.info('Started.')


    '''
        Receives a frame: a network delimiter and the following payload described by the delimiter
        This method always finishes the frame reading
    '''
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
            # 5th component: 32-bit padding to make the delimiter to have 16 bytes
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
    
    '''
        The run loop
    '''
    def _runLoop(self):
        # Connect to the host and reconnect until it has been set not to wantActive
        try:
            while self.wantActive:
                self.active = False
                self.connected = False
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                try:
                    logger.info('Connecting {}:{}...'.format(self.address, self.port))
                    self.socket.connect((self.address, self.port))
                except:
                    t = 30
                    while t > 0 and self.wantActive:
                        if self.verbose > 1 and t % 10 == 0:
                            dots = '.' * int(4 - t / 10)
                            print('Retry in {:.1f} seconds {}\r'.format(t * 0.1, dots), end='')
                        time.sleep(0.1)
                        t -= 1
                    self.socket.close()
                    continue
                # Keep reading while wantActive
                while self.wantActive:
                    if self._recv() == True:
                        self.relay.handleRadarMessage()
                    else:
                        logger.debug('Server disconnected.')
                        self.relay.handleRadarClose()
                        t = 30
                        while t > 0:
                            if self.verbose > 1 and t % 10 == 0:
                                dots = '.' * int(4 - t / 10)
                                print('Retry in {:.1f} seconds {}\r'.format(t * 0.1, dots), end='')
                            time.sleep(0.1)
                            t -= 1
                        self.socket.close()
                        break;
                    if not self.active:
                        self.active = True
        except KeyboardInterrupt:
            logger.debug('Outside runloop KeyboardInterrupt')
        except:
            logger.debug('Outside runloop', sys.exc_info()[0])
        # Outside of the busy loop
        logger.info('Connection from {} terminated.'.format(self.address))
        self.socket.close()
        self.active = False
        self.socket = None

    def start(self):
        self.wantActive = True
        threading.Thread(target=self._runLoop).start()

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
        
    def close(self):
        if (self.socket):
            self.socket.close()

    def __del__(self):
        self.close()
