import os
import sys

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

def variableInString(name, value):
    if isinstance(value, (bool)):
        return '{}{}{} = {}{}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.purple, value, COLOR.reset)
    elif isinstance(value, (int, float)):
        return '{}{}{} = {}{}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.lime, value, COLOR.reset)
    else:
        return '{}{}{} = {}{}{}'.format(COLOR.orange, name, COLOR.reset, COLOR.yellow, value, COLOR.reset)

def showName():
    rows, columns = os.popen('stty size', 'r').read().split()
    c = int(columns)
    print('Version {}'.format(sys.version_info))
    print(colorize('\n{}\n{}\n{}'.format(' ' * c, 'RadarKit'.center(c, ' '), ' ' * c), COLOR.radarkit))
    print(colorize('\n{}\n{}\n{}'.format(' ' * c, 'PyRadarKit'.center(c, ' '), ' ' * c), COLOR.python) + '\n')
