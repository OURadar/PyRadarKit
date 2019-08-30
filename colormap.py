import colorsys
import numpy as np

def fleximap(count=15, xp=None, cp=None):
    if xp is None and cp is None:
        # Color provided. This array can N x 3 for RGB or N x 4 for RGBA
        cp = [
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
        ]
        # X-axis provided, the number of elements must be N
        xp = [0.0, 0.2, 0.5, 0.8, 1.0]
    # If x is not supplied
    if xp is None:
        xp = np.linspace(0.0, 1.0, len(cp))
    # If color is not supplied
    if cp is None:
        print('Supply xp and cp.')
        return None
    cp = np.array(cp, dtype=float)
    xi = np.linspace(0.0, 1.0, count)
    rgb = np.array([np.interp(xi, xp, cp[:, i]) for i in range(cp.shape[1])]).transpose((1, 0))
    return rgb

# Extended colormap for reflectivity
# s - shades / element (number of shades for blue, green, etc.)
def zmapext(s=3):
    if (s % 3):
        print('Poor choice of {} shades / element. Recommend either 30, 15, 6 or 3.'.format(s))
    count = round(6.6667 * s) + 2
    n = count - 1
    xp = np.zeros(16)
    for i in range(6):
        xp[2 * i + 1] = round(i * s + 1) / n
        xp[2 * i + 2] = round((i + 1) * s) / n
    xp[13] = round(6 * s + 1) / n
    xp[14] = round(6 * s + 0.6667 * s) / n
    xp[15] = 1.0
    cp = [
        [0.00, 0.00, 0.00, 0.0],
        [0.80, 0.60, 0.80, 1.0],    # light purple
        [0.40, 0.20, 0.40, 1.0],    # dark purple
        [0.80, 0.80, 0.60, 1.0],    # light dirty
        [0.40, 0.40, 0.40, 1.0],    # dark gray
        [0.00, 1.00, 1.00, 1.0],    # cyan
        [0.00, 0.00, 1.00, 1.0],    # dark blue
        [0.00, 1.00, 0.00, 1.0],    # light green
        [0.00, 0.50, 0.00, 1.0],    # dark green
        [1.00, 1.00, 0.00, 1.0],    # yellow
        [1.00, 0.50, 0.00, 1.0],    # orange
        [1.00, 0.00, 0.00, 1.0],    # torch red
        [0.50, 0.00, 0.00, 1.0],    # dark red
        [1.00, 0.00, 1.00, 1.0],    # magenta
        [0.56, 0.35, 1.00, 1.0],    # purple
        [1.00, 1.00, 1.00, 1.0]     # white
         ]
    return fleximap(count, xp, cp)

# Standard colormap for reflectivity
def zmapstd(s=3):
    rgba = zmapext(s)
    rgba = np.concatenate(([[0.00, 0.00, 0.00, 0.0]], rgba[2 * s + 1:,]))
    return rgba

def zmap():
    # The body shades starts with cyan at index 74 (75th shade)
    # color[ 74] should be (0, 1, 1) cyan at exactly 5 dBZ    (RadarKit)
    # color[104] should be (0, 1, 0) green at exactly 20 dBZ  (RadarKit)
    zero = np.zeros((4, 4))
    head = fleximap(7, [0.0, 0.5, 1.0], [[0.00, 0.00, 0.00, 0.3765],
                                         [0.25, 0.30, 0.35, 1.0000],
                                         [0.50, 0.60, 0.70, 1.0000]])
    head = np.repeat(np.expand_dims(head, axis=1), 10, axis=1).reshape(70, 4)
    body = zmapstd()[1:-1]
    body = np.repeat(np.expand_dims(body, axis=1), 10, axis=1).reshape(140, 4)
    tail = fleximap(256 - 214, [0.0, 1.0], [[1.00, 1.00, 1.00, 1.00], [0.50, 0.50, 0.50, 0.50]])
    return np.concatenate((zero, head, body, tail))

def zmapx():
    zero = np.zeros((14, 4))
    body = zmapext()[1:-1]
    body = np.repeat(np.expand_dims(body, axis=1), 10, axis=1).reshape(200, 4)
    tail = fleximap(256 - 214, [0.0, 1.0], [[1.00, 1.00, 1.00, 1.00], [0.50, 0.50, 0.50, 0.50]])
    return np.concatenate((zero, body, tail))

# Red green map for velocity
def rgmap(count=16):
    xp = [0.0, 0.3, 0.5, 0.7, 1.0]
    cp = [
        [0.00, 0.20, 0.00],
        [0.00, 0.80, 0.00],
        [0.85, 0.85, 0.85],
        [0.80, 0.00, 0.00],
        [0.20, 0.00, 0.00]
    ]
    return fleximap(count, xp, cp)

# Red green map with forced middle 3 shades
def rgmapf(count=16):
    m = count - 1
    c = np.floor(count / 2)
    xp = [0.0, (c - 2) / m, (c - 1) / m, c / m, (c + 1) / m, (c + 2) / m, 1.0]
    cp = [
        [0.00, 1.00, 0.00],
        [0.00, 0.40, 0.00],
        [0.22, 0.33, 0.22],
        [0.40, 0.40, 0.40],
        [0.33, 0.22, 0.22],
        [0.45, 0.00, 0.00],
        [1.00, 0.00, 0.00]
    ]
    return fleximap(count, xp, cp)

def vmap():
    return rgmap()

def wmap(s=4):
    if s % 2:
        print('Poor choice of {} shades / element. Recommend either 2, 4, 8 or 16.'.format(s))
    rgba = np.concatenate((
        fleximap(s, [0.0, 1.0], [[0.00, 1.00, 1.00, 1.00], [0.00, 0.00, 0.85, 1.00]]),
        fleximap(s, [0.0, 1.0], [[0.00, 0.50, 0.00, 1.00], [0.00, 1.00, 0.00, 1.00]]),
        fleximap(s, [0.0, 1.0], [[1.00, 1.00, 0.00, 1.00], [1.00, 0.50, 0.00, 1.00]]),
        fleximap(s, [0.0, 1.0], [[1.00, 0.00, 0.00, 1.00], [0.50, 0.00, 0.00, 1.00]]),
        fleximap(s, [0.0, 1.0], [[1.00, 0.00, 1.00, 1.00], [0.50, 0.00, 0.50, 1.00]]),
        fleximap(s, [0.0, 1.0], [[0.60, 0.22, 1.00, 1.00], [0.35, 0.11, 0.55, 1.00]])
    ))
    rgba = np.repeat(np.expand_dims(rgba, axis=1), 10, axis=1).reshape(s * 6 * 10, 4)
    tail = fleximap(256 - s * 6 * 10, [0.0, 1.0], [[0.70, 0.70, 0.70, 0.70], [0.50, 0.50, 0.50, 0.50]])
    #np.tile([0.20, 0.45, 0.60], int(s / 2)).reshape(-1, 3),
    return np.concatenate((rgba, tail))

def dmap():
    xp = [0.00,
          9.0 / 254.0,
         10.0 / 254.0,
         39.0 / 254.0,
         40.0 / 254.0,
         69.0 / 254.0,
         70.0 / 254.0,
         99.0 / 254.0,
        100.0 / 254.0,
        129.0 / 254.0,
        130.0 / 254.0,
        159.0 / 254.0,
        160.0 / 254.0,
        189.0 / 254.0,
        190.0 / 254.0,
        219.0 / 254.0,
        220.0 / 254.0,
        249.0 / 254.0,
        250.0 / 254.0,
        1.00];
    cp = [
        [0.30, 0.45, 0.50],    #
        [0.60, 0.90, 1.00],    #
        [0.45, 0.20, 0.80],    #
        [0.70, 0.40, 1.00],    #
        [0.50, 0.20, 0.35],    #
        [1.00, 0.50, 0.85],    #
        [0.70, 0.50, 0.15],    #
        [1.00, 1.00, 0.85],    #
        [1.00, 1.00, 1.00],    # 0dB
        [0.00, 0.35, 1.00],    #
        [0.10, 1.00, 0.50],    # 3dB
        [0.00, 0.50, 0.00],    #
        [1.00, 1.00, 0.00],    # 6dB
        [1.00, 0.50, 0.00],    #
        [1.00, 0.00, 0.00],    #
        [0.50, 0.00, 0.00],    #
        [1.00, 0.00, 1.00],    #
        [0.50, 0.00, 0.50],    #
        [1.00, 1.00, 1.00],    #
        [0.60, 1.00, 1.00]     #
    ]
    rgb = fleximap(51, xp, cp)
    rgb = np.repeat(np.expand_dims(rgb, axis=1), 5, axis=1).reshape(5 * 51, 3)
    rgb = np.concatenate((rgb, rgb[-1, :].reshape(1, 3)))
    rgba = np.concatenate((rgb, np.ones((256, 1))), axis=1)
    rgba[:11, 3] = 220.0 / 255.0
    rgba[-6:, 3] = 220.0 / 255.0
    return rgba

def pmap():
    rgb = zebra(64, b=4)
    rgb = np.expand_dims(rgb, axis=2)
    rgb = np.repeat(rgb, 4, axis=2).transpose((0, 2, 1)).reshape(256, 3)
    rgba = np.concatenate((rgb, np.ones((256, 1))), axis=1)
    return rgba

def rmap():
    c = 6
    lomap = fleximap(7, [0.0, 1.0], [[0.00, 0.00, 0.00, 0.00], [0.50, 0.60, 0.70, 1.0]])
    himap = np.concatenate((
        fleximap(5, [0.0, 1.0], [[0.00, 1.00, 1.00, 1.00], [0.00, 0.00, 0.85, 1.00]]),
        fleximap(5, [0.0, 1.0], [[0.00, 1.00, 0.00, 1.00], [0.00, 0.50, 0.00, 1.00]]),
        fleximap(5, [0.0, 1.0], [[1.00, 1.00, 0.00, 1.00], [1.00, 0.50, 0.00, 1.00]]),
        fleximap(5, [0.0, 1.0], [[1.00, 0.00, 0.00, 1.00], [0.50, 0.00, 0.00, 1.00]]),
        fleximap(5, [0.0, 1.0], [[1.00, 0.00, 1.00, 1.00], [0.50, 0.00, 0.50, 1.00]]),
        fleximap(5, [0.0, 1.0], [[0.60, 0.22, 1.00, 1.00], [0.35, 0.11, 0.55, 1.00]]),
        fleximap(5, [0.0, 1.0], [[0.40, 0.45, 1.00, 1.00], [0.20, 0.22, 0.60, 1.00]])
    ))
    n = 256 - c * (lomap.shape[0] + himap.shape[0])
    rgba = np.zeros((256, 4))
    rgba[n:] = np.concatenate((
        np.repeat(np.expand_dims(lomap, axis=1), c, axis=1).reshape((7 * c, 4)),
        np.repeat(np.expand_dims(himap, axis=1), c, axis=1).reshape((5 * 7 * c, 4))
    ))
    return rgba

def kmap():
    # Four bands in the middle, two tail ends
    s = 10
    t = (256 - 6 * s * 4) / 4 / 2;
    rgba = np.concatenate((
        fleximap(s + t, [0.0, 1.0], [[0.35, 0.15, 0.60, 1.00], [0.75, 0.45, 1.00, 1.00]]),
        fleximap(s    , [0.0, 1.0], [[0.50, 0.20, 0.35, 1.00], [1.00, 0.50, 0.85, 1.00]]),
        fleximap(s    , [0.0, 1.0], [[0.70, 0.50, 0.15, 1.00], [1.00, 1.00, 0.85, 1.00]]),
        fleximap(s    , [0.0, 1.0], [[1.00, 1.00, 1.00, 1.00], [0.00, 0.35, 1.00, 1.00]]),
        fleximap(s    , [0.0, 1.0], [[0.20, 1.00, 0.00, 1.00], [0.00, 0.50, 0.00, 1.00]]),
        fleximap(s + t, [0.0, 1.0], [[0.40, 0.45, 1.00, 1.00], [0.20, 0.22, 0.60, 1.00]])
    ))
    # Repeat each color 4 times
    rgba = np.repeat(np.expand_dims(rgba, axis=1), 4, axis=1).reshape(256, 4)
    return rgba

# From reference:
# Hooker, S. B. et al, Detecting Dipole Ring Separatrices with Zebra
# Palettes, IEEE Transactions on Geosciences and Remote Sensing, vol. 33,
# 1306-1312, 1995
def zebra(n=256, b=4, m=0.5):
    x = np.arange(n)
    saw = np.mod(b * x, b)
    hue = 0.999 * np.exp(-3.0 * x / (n - 1))
    sat = m + (1.0 - m) * 0.5 * (1.0 + saw / (b - 1.0))
    val = m + (1.0 - m) * 0.5 * (1.0 + np.cos(4.0 * b * np.pi * x / n))
    return [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hue, sat, val)]
