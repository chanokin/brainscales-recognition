from __future__ import (print_function,
                        unicode_literals,
                        division)
from builtins import str, open, range, dict

import numpy as np
import copy
import os
from .wafer_blacklists import BLACKLISTS


def compute_coords(row_widths, row_starts):
    coords = {}
    coords['i2c'] = {}
    coords['c2i'] = {}
    id_start = 0
    for i in range(len(row_widths)):
        id_start += 0 if i == 0 else row_widths[i - 1]
        width = row_widths[i]
        start = row_starts[i]
        coords[i] = start + np.arange(width)
        ids = id_start + np.arange(width)
        coords['i2c'].update({ids[j]: (i, coords[i][j]) for j in range(width)})
        coords['c2i'][i] = {coords[i][j]: ids[j] for j in range(width)}

    return coords


def compute_row_id_starts(row_widths):
    starts = np.roll(np.cumsum(row_widths), 1)
    starts[0] = 0
    starts.flags.writeable = False
    return starts


def compute_row_ranges(row_widths):
    '''
    per row = (starting id [inclusive], ending id [exclusive])
    so we can do range[0] <= x < range[1] which should help with slices
    '''
    ranges = []
    s = 0
    for w in row_widths:
        ranges.append((s, s + w))
        s += w
    return tuple(map(tuple, ranges))


WAFER_ROW_WIDTHS = np.array([12, 12, 20, 20, 28, 28, 36, 36, 36, 36, 28, 28, 20, 20, 12, 12])
WAFER_ROW_WIDTHS.flags.writeable = False
WAFER_ROW_X_STARTS = np.array([12, 12, 8, 8, 4, 4, 0, 0, 0, 0, 4, 4, 8, 8, 12, 12])
WAFER_ROW_X_STARTS.flags.writeable = False
WAFER_ROW_ID_STARTS = compute_row_id_starts(WAFER_ROW_WIDTHS)
WAFER_ROW_RANGES = compute_row_ranges(WAFER_ROW_WIDTHS)
WAFER_ROW_COORDS = compute_coords(WAFER_ROW_WIDTHS, WAFER_ROW_X_STARTS)

WAFER_CHIPS_PER_RETICLE = 8
WAFER_CHIPS_SHAPE = (2, 4)
WAFER_RETICLE_ROW_WIDTHS = np.array([3, 5, 7, 9, 9, 7, 5, 3])
WAFER_RETICLE_ROW_WIDTHS.flags.writeable = False
WAFER_RETICLE_ROW_X_STARTS = np.array([3, 2, 1, 0, 0, 1, 2, 3])
WAFER_RETICLE_ROW_X_STARTS.flags.writeable = False


class Wafer(object):
    _row_widths = WAFER_ROW_WIDTHS
    _row_x_starts = WAFER_ROW_X_STARTS
    _row_id_starts = WAFER_ROW_ID_STARTS
    _row_ranges = WAFER_ROW_RANGES
    _row_coords = WAFER_ROW_COORDS
    _width = 36  ### max(row_widths)
    _height = 16  ### rows per wafer
    _reticle_row_widths = WAFER_RETICLE_ROW_WIDTHS
    _reticle_row_x_starts = WAFER_RETICLE_ROW_X_STARTS

    def __init__(self, wafer_id=33):
        self._id = wafer_id
        self._used_chips = [{} for _ in range(self._height)]
        self._blacklist = BLACKLISTS[self._id]
        self._get_distances()

    def _get_distances(self):
        cwd = os.path.dirname( os.path.realpath(__file__) )
        dist_file = os.path.join(cwd, 'wafer_distance_cache.npz')
        # if os.path.isfile(dist_file):
        #     # print(np.load(dist_file).keys())
        #     loaded = np.load(dist_file)
        #     self.distances = loaded['distances']
        #     self.id2idx = loaded['id2idx'].item()
        #     return

        id2idx = {}
        ids, coords = self.available()
        n_avail = len(ids)
        distances = np.zeros((n_avail, n_avail))
        for i in range(n_avail):
            id2idx[ids[i]] = i
            for j in range(n_avail):
                if i == j:
                    distances[i, j] = 1000.0 #big number to avoid NaN later
                else:
                    ri, ci = coords[i]
                    rj, cj = coords[j]
                    distances[i, j] = np.sqrt((ri - rj)**2 + (ci - cj)**2)

        np.savez_compressed(dist_file,
            distances=distances, id2idx=id2idx)
        self.distances = distances
        self.id2idx = id2idx


    def available(self):
        ids = []
        coords = []
        for row in range(self._height):
            for col in sorted(self._row_coords[row]):
                id = self._row_coords['c2i'][row][col]
                if id not in self._used_chips[row] and \
                    id not in self._blacklist[row]:
                    ids.append(id)
                    coords.append((row, col))

        return ids, coords

    def _in_range(self, chip_id, id_range):
        return (id_range[0] <= chip_id < id_range[1])

    def find_row(self, chip_id):
        '''Find wafer row given a chip id. To locate the row, since we don't
        have a nice rectangular grid, we need to locate when the difference
        between row id start and the chip id changes from negative to
        positive or it is zero.
        '''
        diffs = np.array(self._row_id_starts) - chip_id
        whr = np.where(diffs <= 0)[0]
        ### last element is always the closest to 0 (inclusive) thus
        ### the correct row
        return whr[-1]

    def insert(self, pop_id, chip_id):
        row = self.find_row(chip_id)
        if chip_id in self._used_chips[row]:
            print('This chip was alread used, try another one')
            return False
        if chip_id in self._blacklist[row]:
            print('This chip was blacklisted, try another one')
            return False

        self._used_chips[row][chip_id] = pop_id
        return True

    def remove(self, chip_id):
        row = self.find_row(chip_id)
        try:
            del self._used_chips[row][chip_id]
            return True
        except:
            print('Chip ({}) not not found in register'.format(chip_id))
            return False

    def render(self, fig_width=15):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        fig = plt.figure(figsize=(fig_width, fig_width))

        ax = plt.subplot(1, 1, 1)

        for row in self._row_coords['c2i']:
            for col in self._row_coords['c2i'][row]:
                _id = self._row_coords['c2i'][row][col]
                if _id in self._used_chips[row]:
                    color = 'blue'
                elif _id in self._blacklist[row]:
                    color = 'red'
                else:
                    color = 'green'

                plt.text(col, row, '{:3d}'.format(_id),
                         horizontalalignment='center', verticalalignment='center',
                         bbox=dict(  # xy=(col-0.5, row-0.5), width=1., height=1.,
                             facecolor=color, edgecolor='none', alpha=0.333)
                         )

        ax.set_xlim(-1, self._width + 1)
        ax.set_ylim(self._height + 1, -1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.tight_layout()
        plt.savefig('wafer_render.pdf')
        plt.close(fig)
        print('Render saved to {}'.format(os.path.join(os.getcwd(), 'wafer_render.pdf')))


    def clone(self):
        wfr = Wafer(self._id)
        wfr._used_chips[:] = [copy.copy(used) for used in self._used_chips]
        return wfr

    def i2c(self, i):
        return self._row_coords['i2c'][i]

    def c2i(self, r, c):
        return self._row_coords['c2i'][r][c]

    def c2i(self, coord):
        return self.c2i(coord[0], coord[1])