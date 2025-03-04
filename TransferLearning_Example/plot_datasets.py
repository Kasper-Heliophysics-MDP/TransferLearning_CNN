#!/usr/bin/env python

import itertools
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from multiprocessing import Pool, cpu_count

from astropy.time import Time

from data_gen_test import load_all_files, tile_data, DataGen

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

number_of_cores = cpu_count()#int(os.environ['SLURM_CPUS_PER_TASK']) 
print(number_of_cores)
backbone = 'resnet34'
batch_size = 32
# AUTOTUNE = tf.data.AUTOTUNE
window_size = 256

root_data_dir = pathlib.Path("/data/pmurphy/")
tf.keras.backend.clear_session()
num_classes=2

np_files = list(root_data_dir.glob('SUN_TRACKING*0'))
ltime = load_all_files('time.npy')
lfreq = load_all_files('frequency.npy')
lmask = load_all_files('mask.npy')
ldata = load_all_files('data.npy')

r_times, r_freqs = tile_data(ltime, lfreq, window_size)
r_times = Time(r_times)

# Need to do proper stratified sampling
rng = np.random.default_rng(seed=42)
shuff_ind = rng.permutation(len(r_times))
r_times = r_times[shuff_ind]
r_freqs = r_freqs[shuff_ind]
train_val_ratio = 0.8

r_times_val = r_times[int(len(r_times)*train_val_ratio):]
r_freqs_val = r_freqs[int(len(r_freqs)*train_val_ratio):]

r_times_test = r_times_val[len(r_times_val)//2:]
r_freqs_test = r_freqs_val[len(r_freqs_val)//2:]

r_times_val = r_times_val[:len(r_times_val)//2]
r_freqs_val = r_freqs_val[:len(r_freqs_val)//2]

r_times = r_times[:int(len(r_times)*train_val_ratio)]
r_freqs = r_freqs[:int(len(r_freqs)*train_val_ratio)]


train_gen = DataGen(r_times,
                    r_freqs,
                    batch_size,
                    shuffle=False,
                    window_size=window_size,
                    backbone=backbone)

val_gen = DataGen(r_times_val,
                r_freqs_val,
                batch_size,
                shuffle=False,
                window_size=window_size,
                backbone=backbone)

test_gen = DataGen(r_times_test,
                r_freqs_test,
                batch_size,
                shuffle=False,
                window_size=window_size,
                backbone=backbone)

# train_gen.__plotitem__(0,"train")
with Pool() as pool:
    print("Plotting {} train images".format(train_gen.__len__()))
    pool.starmap(train_gen.__plotitem__, zip(range(12561, train_gen.__len__()), itertools.repeat("train")))
    print("Plotting {} val images".format(val_gen.__len__()))
    pool.starmap(val_gen.__plotitem__, zip(range(val_gen.__len__()), itertools.repeat("val")))
    print("Plotting {} test images".format(test_gen.__len__()))
    pool.starmap(test_gen.__plotitem__, zip(range(test_gen.__len__()), itertools.repeat("test")))

# for i in range(train_gen.__len__()):
#     train_gen.__plotitem__(i, "train")

# for i in range(val_gen.__len__()):
#     val_gen.__plotitem__(i, "val")

# for i in range(test_gen.__len__()):
#     test_gen.__plotitem__(i, "test")