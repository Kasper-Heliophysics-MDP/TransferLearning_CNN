#!/usr/bin/env ptyhon

import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from pathlib import Path

from astropy.time import Time
from matplotlib import dates

# from data_gen_test import load_all_files

def load_all_files(fname):
    loaded_files = np.array([np.load(np_f/fname, mmap_mode='r')
                             for np_f in np_files])
    arr = []
    for lf in loaded_files:
        arr.append(lf)
    return arr


def plot_date(data, times, freqs, plot_type="data"):
    date_format = dates.DateFormatter("%H:%M:%S")
    times = Time(times)
    freqs = freqs/1e6
    fig, ax = plt.subplots(data.shape[2],1, sharex=True, sharey=True, figsize=(16,12))
    cmap_list = ['viridis',
                 'inferno',
                 'Purples',
                 'Oranges',
                 'Greens',
                 'Blues']
    subplot_title_list = ['Stokes I',
                          'Absolute Stokes V',
                          'Type II',
                          'Type III',
                          'Type IV',
                          'Continuous Emission']
    for i in range(data.shape[2]):
        im = ax[i].imshow(data[:,:,i].T,
                     aspect='auto',
                     origin='lower',
                     cmap=cmap_list[i],
                     extent=[times[0].plot_date,
                             times[-1].plot_date,
                             freqs[0],
                             freqs[-1]
                     ],
                     )
        ax[i].set_title(subplot_title_list[i])
        fig.colorbar(im, ax=ax[i])
    ax[-1].xaxis_date()
    ax[-1].xaxis.set_major_formatter(date_format)
    fig.supylabel("Frequency (MHz)")
    fig.supxlabel("Time on {}".format(times[0].isot[:10]))
    plt.tight_layout()
    print("saving")
    plt.savefig("/data/pmurphy/dataset_plots/{}_{}.png".format(times[0].isot[:10], plot_type))
    plt.close()
    return

print("loading")
root_data_dir = Path("/minerva/pmurphy/")
np_files = list(root_data_dir.glob('SUN_TRACKING*0'))
np_files.sort()
ltime = load_all_files('time.npy')
lfreq = load_all_files('frequency.npy')
lmask = load_all_files('mask.npy')
ldata = load_all_files('data_rescale.npy')
print("plotting")
with Pool() as pool:
    pool.starmap(plot_date,
             zip(ldata, ltime, lfreq, itertools.repeat("data")))
    pool.starmap(plot_date,
             zip(lmask, ltime, lfreq, itertools.repeat("mask")))