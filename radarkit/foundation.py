import os
import sys
import time
import threading

from ._version import __version__
from .rk import *

if 'b' in version():
    __version__ += 'b'
    version_info = __version__

# Some color escape codes for pretty strings in terminal
class COLOR:
    reset = "\033[0m"
    red = "\033[38;5;196m"
    orange = "\033[38;5;214m"
    yellow = "\033[38;5;226m"
    lime = "\033[38;5;118m"
    green = "\033[38;5;46m"
    teal = "\033[38;5;49m"
    iceblue = "\033[38;5;51m"
    skyblue = "\033[38;5;45m"
    blue = "\033[38;5;27m"
    purple = "\033[38;5;99m"
    indigo = "\033[38;5;201m"
    hotpink = "\033[38;5;199m"
    pink = "\033[38;5;213m"
    deeppink = "\033[38;5;198m"
    salmon = "\033[38;5;210m"
    white = "\033[38;5;15m"
    python = "\033[38;5;226;48;5;24m"
    radarkit = "\033[38;5;15;48;5;124m"
    whiteOnGray = "\033[38;5;15;48;5;241m"


# Constants
class CONSTANTS:
    IP = '127.0.0.1'
    PORT = 10000
    MAX_GATES = 4096
    BUFFER_SIZE = 262144
    PACKET_DELIM_SIZE = 16


# Network packet type according to RadarKit
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


def showName():
    rows, columns = os.popen('stty size', 'r').read().split()
    c = int(columns)
    print('\r{}\n'.format(sys.version_info))
    print(colorize('{}\n{}\n{}'.format(' ' * c, 'Algorithm Manager'.center(c, ' '), ' ' * c), COLOR.whiteOnGray))
    print(colorize('{}\n{}\n{}'.format(' ' * c, 'PyRadarKit {}'.format(version_info).center(c, ' '), ' ' * c), COLOR.python))
    print(colorize('{}\n{}\n{}'.format(' ' * c, 'RadarKit {}'.format(version()).center(c, ' '), ' ' * c), COLOR.radarkit))
    print('')


def showArray(d, letter='U'):
    formatDesc = ' {:6.2f}'
    j = 0
    print('  {} = [ {} ... {} ]'.format(colorize(letter.rjust(3, ' '), COLOR.yellow),
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


def variableInString(name, value):
    if isinstance(value, (bool)):
        return '{}{}{} = {}{}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.purple, value, COLOR.reset)
    elif isinstance(value, int):
        return '{}{}{} = {}{:,}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.lime, value, COLOR.reset)
    elif isinstance(value, float):
        return '{}{}{} = {}{:,.3f}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.lime, value, COLOR.reset)
    else:
        return '{}{}{} = {}{}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.yellow, value, COLOR.reset)


class algorithmRunner(threading.Thread):
    def __init__(self, algorithmQueue, resultQueue, stopper):
        super().__init__(self)
        self.algorithmQueue = algorithmQueue
        self.resultQueue = resultQueue

    def run(self):
        while not self.stopper.is_set():
            try:
                algorithm = self.algorithmQueue.get_nowait()
            except queue.Empty:
                break
            else:
                #result =
                self.resultQueue.put((0, ))
        # Pop a method, then run it


class SignalHandler:
    stopper = None
    workers = None
    def __init__(self, stopper, workers):
        self.stopper = stopper
        self.workers = workers
    def __call__(self, signum, frame):
        self.stopper.set()
        for worker in self.workers:
            worker.join()
        sys.exit(0)
