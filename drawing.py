from __future__ import print_function
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
alphabet_ord = list(map(ord, alphabet))
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
num_to_alpha = dict(enumerate(alphabet_ord))

MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75


def align(coords):
    """
    corrects for global slant/offset in handwriting strokes
    """
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords


def skew(coords, degrees):
    """
    skews strokes by given degrees
    """
    coords = np.copy(coords)
    theta = degrees * np.pi/180
    A = np.array([[np.cos(-theta), 0], [np.sin(-theta), 1]])
    coords[:, :2] = np.dot(coords[:, :2], A)
    return coords


def stretch(coords, x_factor, y_factor):
    """
    stretches strokes along x and y axis
    """
    coords = np.copy(coords)
    coords[:, :2] *= np.array([x_factor, y_factor])
    return coords


def add_noise(coords, scale):
    """
    adds gaussian noise to strokes
    """
    coords = np.copy(coords)
    coords[1:, :2] += np.random.normal(loc=0.0, scale=scale, size=coords[1:, :2].shape)
    return coords


def encode_ascii(ascii_string):
    """
    encodes ascii string to array of ints
    """
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)) + [0])


def denoise(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    coords = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []
    for stroke in coords:
        if len(stroke) != 0:
            x_new = savgol_filter(stroke[:, 0], 7, 3, mode='nearest')
            y_new = savgol_filter(stroke[:, 1], 7, 3, mode='nearest')
            xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])
            stroke = np.concatenate([xy_coords, stroke[:, 2].reshape(-1, 1)], axis=1)
            new_coords.append(stroke)

    coords = np.vstack(new_coords)
    return coords


def interpolate(coords, factor=2):
    """
    interpolates strokes using cubic spline
    """
    coords = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []
    for stroke in coords:

        if len(stroke) == 0:
            continue

        xy_coords = stroke[:, :2]

        if len(stroke) > 3:
            f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='cubic')
            f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='cubic')

            xx = np.linspace(0, len(stroke) - 1, factor*(len(stroke)))
            yy = np.linspace(0, len(stroke) - 1, factor*(len(stroke)))

            x_new = f_x(xx)
            y_new = f_y(yy)

            xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

        stroke_eos = np.zeros([len(xy_coords), 1])
        stroke_eos[-1] = 1.0
        stroke = np.concatenate([xy_coords, stroke_eos], axis=1)
        new_coords.append(stroke)

    coords = np.vstack(new_coords)
    return coords


def normalize(offsets):
    """
    normalizes strokes to median unit norm
    """
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets


def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    offsets = np.concatenate([np.array([[0, 0, 1]]), offsets], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    convert from offsets to coordinates
    """
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def draw(
        offsets,
        ascii_seq=None,
        align_strokes=True,
        denoise_strokes=True,
        interpolation_factor=None,
        save_file=None
):
    strokes = offsets_to_coords(offsets)

    if denoise_strokes:
        strokes = denoise(strokes)

    if interpolation_factor is not None:
        strokes = interpolate(strokes, factor=interpolation_factor)

    if align_strokes:
        strokes[:, :2] = align(strokes[:, :2])

    fig, ax = plt.subplots(figsize=(12, 3))

    stroke = []
    for x, y, eos in strokes:
        stroke.append((x, y))
        if eos == 1:
            coords = zip(*stroke)
            ax.plot(coords[0], coords[1], 'k')
            stroke = []
    if stroke:
        coords = zip(*stroke)
        ax.plot(coords[0], coords[1], 'k')
        stroke = []

    ax.set_xlim(-50, 600)
    ax.set_ylim(-40, 40)

    ax.set_aspect('equal')
    plt.tick_params(
        axis='both',
        left='off',
        top='off',
        right='off',
        bottom='off',
        labelleft='off',
        labeltop='off',
        labelright='off',
        labelbottom='off'
    )

    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
            ascii_seq = ''.join(list(map(chr, ascii_seq)))
        plt.title(ascii_seq)

    if save_file is not None:
        plt.savefig(save_file)
        print('saved to {}'.format(save_file))
    else:
        plt.show()
    plt.close('all')
