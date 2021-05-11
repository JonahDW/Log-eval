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
from matplotlib.lines import Line2D

from math import ceil
from scipy.interpolate import UnivariateSpline

def plot_rms_steps(path, sensitivity, output):
    field = path.split('/')[-3]
    images, data = get_data(path)

    fig, axs = plt.subplots(2,2, figsize=(12,9))
    ax = axs.ravel()

    min_rms = np.log10(np.min([np.min(im) for im in data]))
    max_rms = np.log10(np.min([np.max(im) for im in data]))

    rms_range = np.logspace(min_rms, max_rms, 100)
    for i in range(len(data)):
        rms_im = data[i].flatten()

        coverage = []
        for rms in rms_range:
            coverage.append(np.sum([rms_im < rms])/len(rms_im))

        im_rms = np.median(rms_im)

        data_spline = UnivariateSpline(rms_range, coverage, s=0, k=3)
        data_1d = data_spline.derivative(n=1)

        color = next(ax[0]._get_lines.prop_cycler)['color']

        ax[0].plot(rms_range, coverage, linewidth=1, color=color, label=images[i]['type'])
        ax[0].fill_between(rms_range, 0, coverage, alpha=0.2)

        ax[1].plot(rms_range, data_1d(rms_range), linewidth=1, color=color)

#       ax[2].scatter(i, local_dr, marker='^', color=color)
        ax[3].axhline(float(sensitivity), 0, 1, color='k', ls='--')
        ax[3].plot(i, im_rms, marker='^', color=color)

    ax[0].set_xscale('log')
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].set_xlabel('RMS (Jy)')
    ax[0].set_ylabel('Coverage')

    ax[1].set_xscale('log')
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[1].set_xlabel('RMS (Jy)')
    ax[1].set_ylabel('1st derivative of coverage')

    ax[2].set_xlabel('RMS (Jy)')
    ax[2].set_ylabel('Local dynamic range')

    ax[3].set_yscale('log')
    ax[3].set_ylabel('Mean rms')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.savefig(os.path.join(output,field+'_full_rms.png'), dpi=300)
    plt.close()

def get_data(path):
    images = sorted(glob.glob(os.path.join(path,'*_rms_*.fits')))

    im_index = []
    for im in images:
        image = os.path.basename(im).split('.')[0]
        split_string = image.split('_')
        if len(split_string) == 6:
            im_index.append([im, split_string[-1], 'base'])
        elif len(split_string) == 8:
            im_index.append([im, split_string[-2], split_string[-1]])
        else:
            print('Cannot parse filename!')

    images = Table(np.array(im_index), names=['file','iter','type'])
    images.sort('iter')
    images.reverse()

    images = unique(images, keys='type')

    image_data = []
    for i in images:
        image = fits.open(i['file'])[0].data
        image_data.append(image.flatten())

    return images, image_data