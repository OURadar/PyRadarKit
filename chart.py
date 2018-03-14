import numpy as np
import matplotlib
from scipy.interpolate import griddata

bgColor = (0.89, 0.87, 0.83)

def zmap():
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
    return matplotlib.colors.LinearSegmentedColormap.from_list('zmap', colors, N=len(colors))

class Chart:
    """
        A Chart Class
    """
    def __init__(self, width=6, height=6.5, cmap='default'):
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

        if not cmap is not 'default':
            self.cmap = zmap()
        else:
            self.cmap = zmap()

        self.fig = matplotlib.pyplot.figure(figsize=(self.width, self.height), dpi=144, facecolor=None)
        self.fig.patch.set_alpha(0.0)

        self.ax = matplotlib.pyplot.axes(rect, facecolor=bgColor)

        self.grid_x, self.grid_y = np.mgrid[-50:50:100/(dpi*rect[2]*width), -50:50:100/(dpi*rect[3]*height)]
        a, r = np.mgrid[slice(0, 2.0 * np.pi,  np.pi / 4.0), slice(0, 50, 2.5)]
        x = r * np.sin(a)
        y = r * np.cos(a)
        z = r * 2.0 - 15.0
        points = np.vstack([x.flatten(), y.flatten()]).transpose()
        zi = griddata(points, z.flatten(), (self.grid_x, self.grid_y), method='linear')
        self.im = matplotlib.pyplot.imshow(zi, extent=(-50, 50, -50, 50), vmin=0.0, vmax=80.0, cmap=self.cmap)
        self.ax2 = matplotlib.pyplot.axes(rect, facecolor=None, frameon=False, sharex=self.ax, sharey=self.ax)
        self.xlabel = matplotlib.pyplot.xlabel('X Distance (km)', axes=self.ax2)
        self.ylabel = matplotlib.pyplot.ylabel('Y Distance (km)', axes=self.ax2)
        self.cax = self.fig.add_axes((rect[0], rect[1] + rect[3] + 0.06, rect[2], 0.03))
        self.colorbar = matplotlib.pyplot.colorbar(ax=self.ax2, cax=self.cax, orientation='horizontal')
        self.cax.set_title('Example Reflectivity (dBZ)')

    def set_data(self, x, y, z, extent=None):
        points = np.vstack([x.flatten(), y.flatten()]).transpose()
        if not extent is None:
            dx = (extent[1] - extent[0]) / self.wp
            dy = (extent[3] - extent[2]) / self.hp
            self.grid_x, self.grid_y = np.mgrid[extent[0]:extent[1]:dx, extent[2]:extent[3]:dy]
            self.im.set_extent(extent)
        zi = griddata(points, z.flatten(), (self.grid_x, self.grid_y), method='linear')
        self.im.set_data(zi)

    def set_zdata(z):
        return


def showPPI(x, y, z, cmap='default'):
    w = 6
    h = 6.5
    # Duplicate the first azimuth and append it to the end
    xx = np.append(x, x[0:1, :], axis=0)
    yy = np.append(y, y[0:1, :], axis=0)
    zz = np.append(z, z[0:1, :], axis=0)
    # Now we setup the figure
    fig = matplotlib.pyplot.figure(figsize=(w, h), dpi=144, facecolor=None)
    if w > h:
        rect = [0.14, 0.1, 0.8 * h / w, 0.8]
    else:
        rect = [0.14, 0.1, 0.8, 0.8 * w / h]
    rect = [round(x * 72.0) / 72.0 + 0.5 / 72.0 for x in rect]
    if not cmap is 'default':
        cmap = colormap(cmap)
    else:
        cmap = zmap()
    ax = matplotlib.pyplot.axes(rect, facecolor=bgColor)
    ax.set_xlim((-50, 50))
    ax.set_ylim((-50, 50))
    pc = matplotlib.pyplot.pcolor(xx, yy, zz, vmin=0.0, vmax=80.0, axes=ax, cmap=cmap)
    ax2 = matplotlib.pyplot.axes(rect, facecolor=None, frameon=False, sharex=ax, sharey=ax)
    matplotlib.pyplot.xlabel('X Distance (km)', axes=ax2)
    matplotlib.pyplot.ylabel('Y Distance (km)', axes=ax2)
    # pos = fig.add_axes((0.88, 0.3, 0.03, 0.5))
    cax = fig.add_axes((rect[0], rect[1] + rect[3] + 0.06, rect[2], 0.03))
    cb = matplotlib.pyplot.colorbar(ax=ax2, cax=cax, orientation='horizontal')
    cax.set_title('Reflectivity (dBZ)')
    dic = {'figure':fig, 'axes':ax, 'pcolor':pc, 'coloraxes':cax, 'colobar':cb}
    return dic
