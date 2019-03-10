import sys
import array
import numpy as np
import matplotlib
from scipy.interpolate import griddata

import colormap

bgColor = (0.89, 0.87, 0.83)

matplotlib.rcParams['figure.dpi'] = 144
matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.sans-serif'] = ['System Font',
                                          'Arial',
                                          'DejaVu Sans',
                                          'Bitstream Vera Sans',
                                          'Computer Modern Sans Serif',
                                          'Lucida Grande',
                                          'Verdana',
                                          'Geneva',
                                          'Lucid',
                                          'Arial',
                                          'Helvetica',
                                          'Avant Garde',
                                          'sans-serif']
matplotlib.rcParams['font.serif'] = ['Arial', 
                                     'DejaVu Serif',
                                     'Bitstream Vera Serif',
                                     'Computer Modern Roman',
                                     'New Century Schoolbook',
                                     'Century Schoolbook L',
                                     'Utopia',
                                     'ITC Bookman',
                                     'Bookman',
                                     'Nimbus Roman No9 L',
                                     'Times New Roman',
                                     'Times',
                                     'Palatino',
                                     'Charter',
                                     'serif']
matplotlib.rcParams['text.usetex'] = False


def zmap_local():
    colors = [
        bgColor,
        (0.20, 1.00, 1.00),
        (0.20, 0.60, 1.00),
        (0.00, 0.00, 1.00),
        (0.30, 1.00, 0.00),
        (0.10, 0.80, 0.00),
        (0.00, 0.60, 0.00),
        (1.00, 1.00, 0.00),
        (1.00, 0.75, 0.00),
        (1.00, 0.50, 0.00),
        (1.00, 0.00, 0.00),
        (0.75, 0.00, 0.00),
        (0.50, 0.00, 0.00),
        (1.00, 0.00, 0.80),
        (0.60, 0.30, 1.00),
        (1.00, 1.00, 1.00)
    ]
    return np.array(colors)

def vmap_local():
    return colormap.rgmap(32)

def rho2ind(values):
    m3 = values > 0.93
    m2 = np.logical_and(values > 0.7, ~m3)
    index = values * 52.8751
    index[m2] = values[m2] * 300.0 - 173.0
    index[m3] = values[m3] * 1000.0 - 824.0
    return np.round(index)

class Image:
    """
        A Chart Class
    """
    def __init__(self, z=None, a=None, r=None, extent=[-50, 50, -50, 50], width=5, height=5.5, symbol='Z'):
        dpi = 144
        if width > height:
            rect = [0.14, 0.1, 0.8 * height / width, 0.8]
        else:
            rect = [0.14, 0.1, 0.8, 0.8 * width / height]
        self.width = width
        self.height = height
        self.rect = rect
        self.dpi = dpi
        self.wp = dpi * rect[2] * width
        self.hp = dpi * rect[3] * height

        if symbol is 'K':
            # Not finalized yet
            colors = colormap.kmap()
            vmin = 0.0
            vmax = 0.1 * np.pi
        elif symbol is 'R':
            # Special, does not really matter here
            colors = colormap.rmap()
            vmin = 0.0
            vmax = 256.0
            zz = rho2ind(zz)
        elif symbol is 'P':
            colors = colormap.pmap()
            vmin = -180.0
            vmax = 180.0
        elif symbol is 'D':
            colors = colormap.dmap()
            vmin = -10.0
            vmax = 15.5 + 0.1
        elif symbol is 'W':
            # There is an offset of 1 but okay
            colors = colormap.wmap()
            vmin = 0.0
            vmax = 12.75 + 0.05
        elif symbol is 'V':
            colors = colormap.vmap()
            vmin = -16.0
            vmax = 15.875 + 0.125
        elif symbol is 'Z':
            colors = colormap.zmap()
            d = 0.5
            vmin = -32.0
            vmax = 95.5 + 0.5
        else:
            colors = zmap_local()
            vmin = 0.0
            vmax = 75.0 + 5.0

        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colors', colors[:, :3], N=len(colors))

        self.fig = matplotlib.pyplot.figure(figsize=(self.width, self.height), facecolor=None)
        self.fig.patch.set_alpha(0.0)

        self.ax = matplotlib.pyplot.axes(rect, facecolor=bgColor)

            # dx = (extent[1] - extent[0]) / self.wp
            # dy = (extent[3] - extent[2]) / self.hp
            # self.grid_x, self.grid_y = np.mgrid[extent[0]:extent[1]:dx, extent[2]:extent[3]:dy]

        self.grid_x, self.grid_y = np.mgrid[-50:50:100 / self.wp, -50:50:100 / self.hp]
        if a is None and r is None:
            a, r = np.mgrid[slice(0, 2.0 * np.pi,  np.pi / 180.0), slice(0, 50, 2.5)]
        x = r * np.sin(a)
        y = r * np.cos(a)
        z = r * 2.0 - 15.0
        self.points = np.vstack([x.flatten(), y.flatten()]).transpose()
        zi = griddata(self.points, z.flatten(), (self.grid_x, self.grid_y), method='linear')
        self.im = matplotlib.pyplot.imshow(zi, extent=extent, vmin=vmin, vmax=vmax, cmap=self.cmap)
        self.ax2 = matplotlib.pyplot.axes(rect, facecolor=None, frameon=False, sharex=self.ax, sharey=self.ax)
        self.xlabel = matplotlib.pyplot.xlabel('X Distance (km)', axes=self.ax2)
        self.ylabel = matplotlib.pyplot.ylabel('Y Distance (km)', axes=self.ax2)
        self.cax = self.fig.add_axes((rect[0], rect[1] + rect[3] + 0.06, rect[2], 0.03))
        self.colorbar = matplotlib.pyplot.colorbar(ax=self.ax2, cax=self.cax, orientation='horizontal')
        self.cax.set_title('Example Reflectivity (dBZ)')

    def set_data(self, z, r=None, a=None, symbol='Z', title=None, extent=None):
        if not extent is None:
            dx = (extent[1] - extent[0]) / self.wp
            dy = (extent[3] - extent[2]) / self.hp
            self.grid_x, self.grid_y = np.mgrid[extent[0]:extent[1]:dx, extent[2]:extent[3]:dy]
            self.im.set_extent(extent)
        if not r is None and not a is None:
            x = r * np.sin(a)
            y = r * np.cos(a)
            self.points = np.vstack([x.flatten(), y.flatten()]).transpose()
        zi = griddata(self.points, z.flatten(), (self.grid_x, self.grid_y), method='linear')
        self.im.set_data(zi)
        if symbol is 'K':
            self.colorbar.set_ticks(np.arange(-10, 10, 2))
            if title is None:
                title = 'KDP (degres / km)'
        elif symbol is 'R':
            values = np.array([0.73, 0.83, 0.93, 0.96, 0.99, 1.02, 1.05])
            self.colorbar.set_ticks(rho2ind(values))
            self.colorbar.set_ticklabels(values)
            if title is None:
                title = 'RhoHV (unitless)'
        elif symbol is 'P':
            self.colorbar.set_ticks(np.arange(-180, 181, 60))
            if title is None:
                title = 'PhiDP (degrees)'
        elif symbol is 'D':
            self.colorbar.set_ticks(np.arange(-9, 15, 3))
            if title is None:
                title = 'ZDR (dB)'
        elif symbol is 'W':
            self.colorbar.set_ticks(np.arange(0, 15, 2))
            if title is None:
                title = 'Width (m/s)'
        elif symbol is 'V':
            self.colorbar.set_ticks(np.arange(-15, 16, 3))
            if title is None:
                title = 'Velocity (m/s)'
        elif symbol is 'Z':
            self.colorbar.set_ticks(np.arange(-25, 85, 15))
            if title is None:
                title = 'Reflectivity (dBZ)'
        elif title is None:
            title = 'Data'
        self.cax.set_title(title)

#
#   Chart
#
class Chart:
    def __init__(self, a=None, r=None, values=None, style='S', title=None, maxrange=50.0,
        w=5, h=5.5):
        # When Chart is initialized as Chart(z)
        if not a is None and r is None and values is None:
            values = a
            a = None
            r = None
        # Initialize arrays of coordinates
        if values is None:
            if a is None:
                a = np.arange(360) * np.pi / 180.0
            if r is None:
                r = np.arange(1000) * 60.0
        else:
            if a is None:
                a = np.arange(values.shape[0]) * 2.0 * np.pi / values.shape[0]
            if r is None:
                r = np.arange(values.shape[1]) * 0.06
        rr, aa = np.meshgrid(r, a)
        self.xx = rr * np.sin(aa)
        self.yy = rr * np.cos(aa)

        # Create a new figure
        self.fig = matplotlib.pyplot.figure(figsize=(w, h), dpi=144, facecolor=None)

        # Plot area
        if w > h:
            rect = [0.14, 0.1, 0.8 * h / w, 0.8]
        else:
            rect = [0.14, 0.1, 0.8, 0.8 * w / h]
        rect = np.round(np.array(rect) * 72.0) / 72.0 + 0.5 / 72.0

        # self.ax2 = matplotlib.pyplot.axes(rect, facecolor=None, frameon=False, sharex=self.ax, sharey=self.ax)
        # matplotlib.pyplot.xlabel('X Distance (km)', axes=self.ax2)
        # matplotlib.pyplot.ylabel('Y Distance (km)', axes=self.ax2)

        self.cax = self.fig.add_axes((rect[0], rect[1] + rect[3] + 0.06, rect[2], 0.03))

        self.ax = matplotlib.pyplot.axes(rect, facecolor=bgColor)
        self.ax.set_xlim((-maxrange, maxrange))
        self.ax.set_ylim((-maxrange, maxrange))
        self.ax.set_xlabel('X Distance (km)')
        self.ax.set_ylabel('Y Distance (km)')

        self.pcolormesh = None
        self.colorbar = None

        if values is not None:
            self.set_data(values, style=style, title=title)

    def set_data(self, values, a=None, r=None, style='S', title=None):
        if values is None:
            print('No changes')
            return
        if not a is None and not r is None:
            rr, aa = np.meshgrid(r, a)
            self.xx = rr * np.sin(aa)
            self.yy = rr * np.cos(aa)
        mask = np.isfinite(values)
        # Pick a colormap, vmin, vmax, ticklabels, titlestring, etc. based on style
        if style is 'K':
            # KDP is not finalized yet
            colors = colormap.kmap()
            vmin = 0.0
            vmax = 0.1 * np.pi
            cticks = np.arange(-10, 10, 2)
            cticklabels = None
            titlestring = 'KDP (degres / km)'
        elif style is 'R':
            # Special case, values are mapped to indices
            colors = colormap.rmap()
            vmin = 0.0
            vmax = 256.0
            values = np.copy(values)
            values[mask] = rho2ind(values[mask])
            cticklabels = np.array([0.73, 0.83, 0.93, 0.96, 0.99, 1.02, 1.05])
            cticks = rho2ind(cticklabels)
            titlestring = 'RhoHV (unitless)'
        elif style is 'P':
            colors = colormap.pmap()
            vmin = -180.0
            vmax = 180.0
            cticks = np.arange(-180, 181, 60)
            cticklabels = None
            titlestring = 'PhiDP (degrees)'
        elif style is 'D':
            colors = colormap.dmap()
            vmin = -10.0
            vmax = 15.5 + 0.1
            cticks = np.arange(-9, 15, 3)
            cticklabels = None
            titlestring = 'ZDR (dB)'
        elif style is 'W':
            # I realize there is an offset of 1 but okay
            colors = colormap.wmap()
            vmin = 0.0
            vmax = 12.75 + 0.05
            cticks = np.arange(0, 15, 2)
            cticklabels = None
            titlestring = 'Width (m/s)'
        elif style is 'V':
            # colors = colormap.vmap()
            colors = vmap_local()
            vmin = -16.0
            vmax = 15.875 + 0.125
            cticks = np.arange(-16, 17, 4)
            cticklabels = None
            titlestring = 'Velocity (m/s)'
        elif style is 'Z':
            colors = colormap.zmap()
            d = 0.5
            vmin = -32.0
            vmax = 95.5 + 0.5
            cticklabels = None
            cticks = np.arange(-25, 81, 15)
            titlestring = 'Reflectivity (dBZ)'
        else:
            colors = zmap_local()
            vmin = 0.0
            vmax = 75.0 + 5.0
            cticks = np.arange(-25, 85, 15)
            cticklabels = None
            titlestring = 'Data'
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colors', colors[:, :3], N=len(colors))
        # Keep a copy of the values
        self.values = np.ma.masked_where(~mask, values)
        # Paint the main area
        if self.pcolormesh:
            self.pcolormesh.set_array(self.values[:-1, :-1].ravel())
            self.pcolormesh.set_clim((vmin, vmax))
            self.pcolormesh.set_cmap(cmap)
        else:
            matplotlib.pyplot.sca(self.ax)
            self.pcolormesh = self.ax.pcolormesh(self.xx, self.yy, self.values, cmap=cmap, vmin=vmin, vmax=vmax)
        # Update the colorbar
        if self.colorbar is None:
            self.colorbar = matplotlib.pyplot.colorbar(self.pcolormesh, cax=self.cax, orientation='horizontal')
        # Title, ticks, limits, etc.
        self.colorbar.set_ticks(cticks)
        if not cticklabels is None:
            self.colorbar.set_ticklabels(cticklabels)
        if not title is None:
            titlestring = title
        self.cax.set_title(titlestring)

    def savefig(self, filename):
        self.fig.savefig(filename)

#
#
#

def showPPI(x, y, z, style='S', title=None, maxrange=50.0, dpi=144):
    w = 5
    h = 5.5
    # Duplicate the first azimuth and append it to the end
    xx = np.append(x, x[0:1, :], axis=0)
    yy = np.append(y, y[0:1, :], axis=0)
    zz = np.append(z, z[0:1, :], axis=0)
    mm = ~np.isfinite(zz)
    zz[mm] = 0.0
    # Now we setup the figure
    fig = matplotlib.pyplot.figure(figsize=(w, h), dpi=dpi, facecolor=None)
    if w > h:
        rect = [0.14, 0.1, 0.8 * h / w, 0.8]
    else:
        rect = [0.14, 0.1, 0.8, 0.8 * w / h]
    rect = [round(x * 72.0) / 72.0 + 0.5 / 72.0 for x in rect]
    if style is 'K':
        # Not finalized yet
        colors = colormap.kmap()
        vmin = 0.0
        vmax = 0.1 * np.pi
    elif style is 'R':
        # Special, does not really matter here
        colors = colormap.rmap()
        vmin = 0.0
        vmax = 256.0
        zz = rho2ind(zz)
    elif style is 'P':
        colors = colormap.pmap()
        vmin = -180.0
        vmax = 180.0
    elif style is 'D':
        colors = colormap.dmap()
        vmin = -10.0
        vmax = 15.5 + 0.1
    elif style is 'W':
        # There is an offset of 1 but okay
        colors = colormap.wmap()
        vmin = 0.0
        vmax = 12.75 + 0.05
    elif style is 'V':
        colors = colormap.vmap()
        vmin = -16.0
        vmax = 15.875 + 0.125
    elif style is 'Z':
        colors = colormap.zmap()
        d = 0.5
        vmin = -32.0
        vmax = 95.5 + 0.5
    else:
        colors = zmap_local()
        vmin = 0.0
        vmax = 75.0 + 5.0
    zz = np.ma.masked_where(mm, zz)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('colors', colors[:, :3], N=len(colors))
    ax = matplotlib.pyplot.axes(rect, facecolor=bgColor)
    ax.set_xlim((-maxrange, maxrange))
    ax.set_ylim((-maxrange, maxrange))
    pc = matplotlib.pyplot.pcolormesh(xx, yy, zz, vmin=vmin, vmax=vmax, axes=ax, cmap=cmap)
    ax2 = matplotlib.pyplot.axes(rect, facecolor=None, frameon=False, sharex=ax, sharey=ax)
    matplotlib.pyplot.xlabel('X Distance (km)', axes=ax2)
    matplotlib.pyplot.ylabel('Y Distance (km)', axes=ax2)
    # pos = fig.add_axes((0.88, 0.3, 0.03, 0.5))
    cax = fig.add_axes((rect[0], rect[1] + rect[3] + 0.06, rect[2], 0.03))
    cb = matplotlib.pyplot.colorbar(ax=ax2, cax=cax, orientation='horizontal')
    if style is 'K':
        cb.set_ticks(np.arange(-10, 10, 2))
        if title is None:
            title = 'KDP (degres / km)'
    elif style is 'R':
        values = np.array([0.73, 0.83, 0.93, 0.96, 0.99, 1.02, 1.05])
        cb.set_ticks(rho2ind(values))
        cb.set_ticklabels(values)
        if title is None:
            title = 'RhoHV (unitless)'
    elif style is 'P':
        cb.set_ticks(np.arange(-180, 181, 60))
        if title is None:
            title = 'PhiDP (degrees)'
    elif style is 'D':
        cb.set_ticks(np.arange(-9, 15, 3))
        if title is None:
            title = 'ZDR (dB)'
    elif style is 'W':
        cb.set_ticks(np.arange(0, 15, 2))
        if title is None:
            title = 'Width (m/s)'
    elif style is 'V':
        cb.set_ticks(np.arange(-15, 16, 3))
        if title is None:
            title = 'Velocity (m/s)'
    elif style is 'Z':
        cb.set_ticks(np.arange(-25, 85, 15))
        if title is None:
            title = 'Reflectivity (dBZ)'
    elif title is None:
        title = 'Data'
    cax.set_title(title)
    dic = {'figure':fig, 'axes':ax, 'axesc':ax2, 'pcolor':pc, 'coloraxes':cax, 'colobar':cb}
    return dic

# def updatePPI(ppi, x, y, v, style='S', title=None, maxrange=50.0):
#     ppi['axes'].clear()
#     ppi['coloraxes'].clear()
#     if cmap is None:
#         cmap = zmap()
#     pc = ppi['axes'].pcolormesh(x, y, v, vmin=vmin, vmax=vmax, axes=ppi['axes'], cmap=cmap)
#     cb = matplotlib.pyplot.colorbar(ax=ppi['axesc'], cax=ppi['coloraxes'], orientation='horizontal')
#     if not title is None:
#         ppi['coloraxes'].set_title(title)
