import os
import sys
import glob
import pickle

from sys import argv

import numpy as np
from astropy.io import fits
from astropy.table import Table, unique

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

from math import ceil
from scipy.interpolate import UnivariateSpline

def plot_rms_steps(path, sensitivity, output):
    '''
    Get all rms images for different self calibration steps
    and plot coverage, coverage derivative, dynamic range and rms
    '''
    field = path.split('/')[-3]
    images, data, sources = get_data(path)

    fig, axs = plt.subplots(2,2, figsize=(12,9))
    ax = axs.ravel()

    min_rms = np.log10(np.min([np.min(im) for im in data]))
    max_rms = np.log10(np.min([np.max(im) for im in data]))

    rms_range = np.logspace(min_rms, max_rms, 100)

    # Sort self calibration steps in the correct order
    sorted_steps = sorted(images['type'], key=lambda x: 'b'+x if 'ap' in x else 'a'+x)
    for i, step in enumerate(sorted_steps):
        idx = np.where(images['type'] == step)[0][0]
        rms_im = data[idx].flatten()

        coverage = []
        for rms in rms_range:
            coverage.append(np.sum([rms_im < rms])/len(rms_im))

        bright_idx = np.argpartition(-sources[idx]['Peak_flux'], 5)[:5]
        bright = sources[idx][bright_idx]
        local_dr = np.sum(bright['Peak_flux']/bright['Isl_rms'])

        im_rms = np.median(rms_im)

        data_spline = UnivariateSpline(rms_range, coverage, s=0, k=3)
        data_1d = data_spline.derivative(n=1)

        color = next(ax[0]._get_lines.prop_cycler)['color']

        ax[0].plot(rms_range, coverage, linewidth=1, color=color, label=images[idx]['type'])
        ax[0].fill_between(rms_range, 0, coverage, alpha=0.2)

        ax[1].plot(rms_range, data_1d(rms_range), linewidth=1, color=color)
        ax[2].scatter(i, local_dr, marker='s', color=color)
        ax[3].plot(i, im_rms, marker='s', color=color)

    ax[0].set_xscale('log')
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].set_xlabel('RMS (Jy/beam)')
    ax[0].set_ylabel('Coverage')
    ax[0].set_ylim(bottom=0)

    ax[1].set_xscale('log')
    ax[1].autoscale(enable=True, axis='both', tight=True)
    ax[1].set_xlabel('RMS (Jy/beam)')
    ax[1].set_ylabel('1st derivative of coverage')
    ax[1].set_ylim(bottom=0)
    ax[1].set_yticklabels([])

    coords = np.array([c.get_offsets()[0] for c in ax[2].collections])
    ax[2].plot(coords[:,0],coords[:,1], zorder=0, color='k', ls='--')

    ax[2].set_ylabel('Local dynamic range')
    ax[2].set_xticks(range(len(data)))
    ax[2].set_xticklabels(sorted_steps)
    ax[2].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    coords = np.array([l.get_xydata()[0] for l in ax[3].lines])
    ax[3].plot(coords[:,0],coords[:,1], zorder=0, color='k', ls='--')
    ax[3].axhline(float(sensitivity), 0, 1, color='k', label='Theoretical sensitivity')
    ax[3].legend()

    ax[3].set_ylabel('Mean rms (Jy/beam)')
    ax[3].set_xticks(range(len(data)))
    ax[3].set_xticklabels(sorted_steps)
    ax[3].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.savefig(os.path.join(output,field+'_full_rms.png'), dpi=300)
    plt.close()

def get_data(path):
    images = sorted(glob.glob(os.path.join(path,'*_rms_*.fits')))

    im_index = []
    for im in images:
        cat = im.replace('rms','cat')
        image = os.path.basename(im).split('.')[0]
        split_string = image.split('_')
        if len(split_string) == 6:
            im_index.append([im, cat, split_string[-1], 'base'])
        elif len(split_string) == 8:
            im_index.append([im, cat, split_string[-2], split_string[-1]])
        else:
            print('Cannot parse filename!')

    images = Table(np.array(im_index), names=['imfile','catfile','iter','type'])
    images.sort('iter')
    images.reverse()

    images = unique(images, keys='type')

    image_data = []
    sources = []
    for i in images:
        image = fits.open(i['imfile'])[0].data
        image_data.append(image.flatten())

        catalog = Table.read(i['catfile'])
        sources.append(catalog)

    return images, image_data, sources