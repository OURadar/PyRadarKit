import os
import json
import zipfile
import geopandas
import urllib.request

folder = os.path.expanduser('shapefiles')

def getOverlays(country='United States', verbose=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    subfolder = '{}/{}'.format(folder, country)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Fullpath of the index
    file = '{}/index'.format(subfolder, country)
    if not os.path.isfile(file):
        url = 'https://arrc.ou.edu/~boonleng/shapefiles/{}/index'.format(country)
        print('Downloading {} -> {} ...'.format(url, file))
        urllib.request.urlretrieve(url, file)

    file = '{}/{}/index'.format(folder, country)
    with open(file, 'r') as file:
        data = ''.join(file.read().strip())
    shapes = json.loads(data)

    # Check individual shapes
    overlays = []
    for shape in shapes:
        file = '{}/{}/{}.dbf'.format(folder, country, shape['file'])
        if not os.path.isfile(file):
            url = 'https://arrc.ou.edu/~boonleng/shapefiles/{}/{}.zip'.format(country, shape['file'])
            file = '{}/{}/{}.zip'.format(folder, country, shape['file'])
            if verbose:
                print('Downloading {} -> {} ...'.format(url, file))
            if not os.path.isfile(file):
                urllib.request.urlretrieve(url, file)
            with zipfile.ZipFile(file) as zipped:
                for info in zipped.infolist():
                    file = '{}/{}/{}'.format(folder, country, info.filename)
                    with open(file, 'wb') as outfile:
                        with zipped.open(info) as zippedfile:
                            outfile.write(zippedfile.read())
        # The shapefile
        file = '{}/{}/{}.shp'.format(folder, country, shape['file'])
        df = geopandas.read_file(file)
        if verbose:
            print(file)
        overlays.append(df)
    return overlays
