import os
import sys
import urllib.request

folder = os.path.expanduser('data')
prefix = 'PX-20170220-050706-E2.4'

def check(verbose=1):
    if not os.path.exists(folder):
        print('Making folder {} ...'.format(folder))
        os.makedirs(folder)

    files = ['{}-{}.nc'.format(prefix, x) for x in ['Z', 'V', 'W', 'D', 'P', 'R']]

    # Download data files if not exist
    allExist = True
    for file in files:
        url = 'https://arrc.ou.edu/~boonleng/files/{}'.format(file)
        dst = '{}/{}'.format(folder, file)
        if not os.path.isfile(dst):
            if verbose:
                print('Downloading {} ...'.format(file))
            urllib.request.urlretrieve(url, dst)
            allExist = False
    if allExist and verbose:
        print('Sample data files exist.')

def file():
    check(verbose=0)
    return '{}/{}-Z.nc'.format(folder, prefix)

def files():
    check(verbose=0)
    files = ['{}-{}.nc'.format(prefix, x) for x in ['Z', 'V', 'W', 'D', 'P', 'R']]
    return files
