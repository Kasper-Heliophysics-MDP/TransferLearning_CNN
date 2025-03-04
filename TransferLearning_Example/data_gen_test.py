#!/usr/bin/env python

import argparse
import gc
import os
import itertools
import pathlib
import random
import time

import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf

# from multiprocessing import Pool

from astropy.time import Time
from matplotlib import dates
from tensorflow.keras import layers, losses, models
sm.set_framework('tf.keras')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay, CosineDecayRestarts
from tensorflow.keras.applications.vgg16 import preprocess_input
from segmentation_models import Linknet, Unet, PSPNet, FPN
from segmentation_models import get_preprocessing
from segmentation_models.losses import BinaryCELoss, CategoricalCELoss
from segmentation_models.metrics import IOUScore, FScore
from segmentation_models.utils import set_regularization
from sklearn.model_selection import StratifiedShuffleSplit

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def load_all_files(fname):
    loaded_files = np.array([np.load(np_f/fname, mmap_mode='r')
                             for np_f in np_files])
    arr = []
    for lf in loaded_files:
        arr.append(lf)
    return arr

def tile_data(tarr, farr, window_len=256):
    t_tiles = []
    f_tiles = []
    if type(tarr) is list:
        for i in range(len(tarr)):
            # 2 here because threshold mask induces offset
            t_inds = np.arange(2, len(tarr[i])-window_len, window_len)
            f_inds = np.arange(0, len(farr[i])-window_len, window_len)

            tf_mesh = np.meshgrid(tarr[i][t_inds],farr[i][f_inds])
            t_flat = tf_mesh[0].flatten()
            f_flat = tf_mesh[1].flatten()
            t_tiles.append(t_flat)
            f_tiles.append(f_flat)
    else:
        # 0 here because fake data mask lines up
        t_inds = np.arange(0, len(tarr)-window_len, window_len)
        f_inds = np.arange(0, len(farr)-window_len, window_len)

        tf_mesh = np.meshgrid(tarr[t_inds],farr[f_inds])
        t_flat = tf_mesh[0].flatten()
        f_flat = tf_mesh[1].flatten()
        t_tiles.append(t_flat)
        f_tiles.append(f_flat)
    t_tiles = np.concatenate(t_tiles, axis=0)
    f_tiles = np.concatenate(f_tiles, axis=0)
    return t_tiles, f_tiles

def get_freq_windows(sampled_times, tarr, farr, window_size):
    tarr = load_all_files("time.npy")
    date_list = [t[0][:10] for t in tarr]
    freq_windows = []
    for t in sampled_times:
        date = t[:10]
        d_ind = date_list.index(date)

        f_tile_start = farr[d_ind][::window_size][:-1]
        freq_windows.append(f_tile_start)
    
    return freq_windows

def stratify_sample(tarr, farr, sss, window_size):
    # take stratified K fold sample from input time and frequency data
    # sss is of form 
    # sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    
    t_tiles_train = []
    f_tiles_train = []

    t_tiles_test = []
    f_tiles_test = []

    time_concat = np.concatenate([lt[2::window_size][:-1] for lt in tarr])
    # burst_type_list = np.load("/data/pmurphy/overlap_burst_types.npy", allow_pickle=True)
    # burst_type_concat = np.concatenate([lt[2::window_size][:-1] for lt in burst_type_list])
    k_fold_split = sss.split(time_concat, [t[:10] for t in time_concat])
    # k_fold_split = sss.split(time_concat, burst_type_concat)

    for train_ind, test_ind in k_fold_split:
        train_times = time_concat[train_ind][::50]
        test_times = time_concat[test_ind][::50]


        train_f_tile_start = get_freq_windows(train_times, tarr, farr, window_size)
        test_f_tile_start = get_freq_windows(test_times, tarr, farr, window_size)


        t_tile_train = np.concatenate([np.tile(train_times[i],
                                               len(train_f_tile_start[i])) for i in range(len(train_times))])
        t_tile_test = np.concatenate([np.tile(test_times[i],
                                               len(test_f_tile_start[i])) for i in range(len(test_times))])
        
        f_tile_train = np.concatenate(train_f_tile_start)
        f_tile_test = np.concatenate(test_f_tile_start)

        t_tiles_train.append(t_tile_train)
        t_tiles_test.append(t_tile_test)
        f_tiles_train.append(f_tile_train)
        f_tiles_test.append(f_tile_test)
        
    
    return (t_tiles_train, f_tiles_train), (t_tiles_test, f_tiles_test)

def plot_results(image_batch, mask_batch, layer, model, output_dir, im_index):
    #image_batch, out_batch = next(iter(train_ds))
    #decoded_imgs = conv_ae.predict(image_batch)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df = (200/1024)/8  # MHz
    n_figs = 3
    fig, axs = plt.subplots(n_figs,3,figsize=(10, 10), sharex=True)#,
    #                           gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    # fig.subplots_adjust(wspace=0.02)
    for i in range(n_figs):
        #image_batch, out_batch = next(iter(train_ds))
        decoded_imgs = model.predict(image_batch)
        ypred = tf.cast(decoded_imgs, tf.float64)
        
        IOU = metric_list[1+layer]
        # IOU.threshold = 0.5
        # axs[0,0].set_title("Input Layer {}".format(layer))
        axs[0,1].set_title("Ground Truth Mask")
        axs[0,2].set_title("Predicted Mask")
        im_vmin = np.percentile(image_batch[i][:,:,0],5)
        im_vmax = np.percentile(image_batch[i][:,:,0],95)
        farr = df*np.arange(image_batch[i][:,:,0].shape[0]) + image_batch[i][0,0,2]
        tarr = 0.25*np.arange(image_batch[i][:,:,0].shape[1])
        cmap_list=['viridis', 'inferno', 'Purples', 'Oranges', 'Greens','Blues' ]
        real_im = axs[i,0].imshow(tf.transpose(image_batch[i][:,:,0]),
                                  cmap=cmap_list[0],
                                  aspect='auto',
                                  origin='lower',
                                  extent=[tarr[0], tarr[-1], farr[0], farr[-1]],
                                  vmin=im_vmin,
                                  vmax=im_vmax)
        real_mask = axs[i,1].imshow(tf.transpose(mask_batch[i][:,:,layer]),
                                    cmap=cmap_list[layer],
                                    aspect='auto',
                                    origin='lower',
                                    extent=[tarr[0], tarr[-1], farr[0], farr[-1]],
                                    vmin=0,
                                    vmax=1)
        pred_mask = axs[i,2].imshow(tf.transpose(decoded_imgs[i][:,:,layer]),
                                    cmap=cmap_list[layer],
                                    aspect='auto',
                                    origin='lower',
                                    extent=[tarr[0], tarr[-1], farr[0], farr[-1]],
                                    vmin=0,
                                    vmax=1)
        
                                    #     divider = make_axes_locatable(axs[i,2])
                                    #     cax1 = divider.append_axes("right", size="5%", pad=0.05)
                                    #     cax1 = fig.add_axes([0, 0, 0.1, 1])
        fig.colorbar(real_im, ax=axs[i,0])
        fig.colorbar(real_mask, ax=axs[i,1])
        fig.colorbar(pred_mask, ax=axs[i,2])
        start_time = Time(image_batch[i][0,0,3], format='mjd').isot
        axs[i,0].set_title("Start Time {}".format(start_time))
        # axs[i,0].axis("off")
        # axs[i,1].axis("off")
        # axs[i,2].axis("off")
        fig.supylabel("Frequency (MHz)")
        fig.supxlabel("Seconds")
        # print(str(np.round(iou, 2)))
        iou = IOU(mask_batch[np.newaxis, i], ypred[np.newaxis, i]).numpy()
        axs[i,2].text(tarr[(128//256)*256],
                      farr[(220//256)*256],
                      "IOU: {}".format(str(np.round(iou, 2))),
                      fontsize=14,
                      backgroundcolor='black',
                      color='white')
        if np.max(decoded_imgs[i][:,:,layer]) >= 0.8:
            T, F = np.meshgrid(tarr, farr)
            # F = np.flip(F, axis=0)
            axs[i,0].contour(T, F, tf.transpose(decoded_imgs[i][:,:,layer]), [0.8], colors='white')
    outfile = output_dir+"layer_{}_compare_{}.png".format(layer, str(im_index).zfill(3))
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close('all')

def create_train_test_val_ds(tarr, farr, sss, window_size, k_fold,
                             batch_size=32, backbone='resnet34', n_channels=4, spikes=False):
    # create tf datasets for a given k_fold
    if not spikes:
        (r_times, r_freqs), (r_times_val, r_freqs_val) = stratify_sample(tarr,
                                                                        farr,
                                                                        sss,
                                                                        window_size
                                                                        )
        # print(r_times.shape, r_times_val.shape, r_freqs.shape, r_freqs_val.shape)
        r_times = Time(r_times[k_fold])
        r_times_val = Time(r_times_val[k_fold])
        
        r_freqs = r_freqs[k_fold]
        r_freqs_val = r_freqs_val[k_fold]
        
        # r_times_test = r_times_val[len(r_times_val)//2:]
        # r_freqs_test = r_freqs_val[len(r_freqs_val)//2:]

        # r_times_val = r_times_val[:len(r_times_val)//2]
        # r_freqs_val = r_freqs_val[:len(r_freqs_val)//2]

        # print(r_times.shape, r_times_val.shape, r_freqs.shape, r_freqs_val.shape)
    else:
        spike_times = Time(np.load("/data/pmurphy/spike_times.npy", allow_pickle=True)) - (window_size/2)*0.25*u.s
        spike_freqs = np.load("/data/pmurphy/spike_freqs.npy", allow_pickle=True) - ((200e6/1024)/8)*(window_size/2)
        rng = np.random.default_rng(seed=42)
        shuff_ind = rng.permutation(len(spike_times))
        r_times = spike_times[shuff_ind]
        r_freqs = spike_freqs[shuff_ind]
        train_val_ratio = 0.7

        r_times_val = r_times[int(len(r_times)*train_val_ratio):]
        r_freqs_val = r_freqs[int(len(r_freqs)*train_val_ratio):]

        r_times = r_times[:int(len(r_times)*train_val_ratio)]
        r_freqs = r_freqs[:int(len(r_freqs)*train_val_ratio)]
    
    train_gen = DataGen(r_times,
                        r_freqs,
                        batch_size,
                        shuffle=False,
                        window_size=window_size,
                        backbone=backbone,
                        n_channels=n_channels,
                        num_classes = num_classes,
                        augment=True,
                        aug_len=25)

    val_gen = DataGen(r_times_val,
                    r_freqs_val,
                    batch_size,
                    shuffle=False,
                    window_size=window_size,
                    n_channels=n_channels,
                    num_classes = num_classes,
                    backbone=backbone)

    # test_gen = DataGen(r_times_test,
    #                 r_freqs_test,
    #                 batch_size,
    #                 shuffle=False,
    #                 window_size=window_size,
    #                 n_channels=n_channels,
    #                 backbone=backbone)

    train_ds = tf.data.Dataset.from_generator(train_gen,
                                            output_signature=(
                                            tf.TensorSpec(shape=(window_size,window_size,n_channels),
                                                            dtype=tf.float64),
                                            tf.TensorSpec(shape=(window_size,window_size,num_classes),
                                                            dtype=tf.float64)))

    val_ds = tf.data.Dataset.from_generator(val_gen,
                                            output_signature=(
                                            tf.TensorSpec(shape=(window_size,window_size,n_channels),
                                                        dtype=tf.float64),
                                            tf.TensorSpec(shape=(window_size,window_size,num_classes),
                                                        dtype=tf.float64)))

    # test_ds = tf.data.Dataset.from_generator(test_gen,
    #                                         output_signature=(
    #                                         tf.TensorSpec(shape=(window_size,window_size,n_channels),
    #                                                     dtype=tf.float64),
    #                                         tf.TensorSpec(shape=(window_size,window_size,num_classes),
    #                                                     dtype=tf.float64)))


    def filter_zeros(X, y):
        if tf.math.reduce_sum(X) !=0:
            return True
        else:
            return False
    # train_ds = tf.data.Dataset.from_tensor_slices(list(train_ds))
    # val_ds = tf.data.Dataset.from_tensor_slices(list(val_ds))


    train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(train_gen.__len__())) 
    val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(val_gen.__len__()))

    # print("original len", train_gen.__len__())

    # train_ds = train_ds.filter(filter_zeros)
    # val_ds = val_ds.filter(filter_zeros)
    # # for X,y in train_ds:
    # #     print(tf.math.reduce_sum(X))
    # train_filter_len = len(list(train_ds))
    # print("Train filter len:", train_filter_len)
    # train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(train_filter_len))

    # val_filter_len = len(list(val_ds))
    # val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(val_filter_len))
    # test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(test_gen.__len__()))
    

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    # # test_ds = configure_for_performance(test_ds)
    

    return train_ds, val_ds

class DataGen:
    def __init__(self,
                 t_pointer,
                 f_pointer,
                 batch_size,
                 window_size=256,
                 shuffle=False,
                 backbone='vgg16',
                 n_channels=3,
                 num_classes=2,
                 augment=False,
                 aug_len=1):
        """
        t_pointer: array of ~astropy.time.Time~ for start of window
                    shape (N)
        f_pointer:  array start frequencys of window
        batch_size = batch size
        shuffle = boolean, shuffle after each epoch doesn't actually make sense right now

        n_channels = number of channels, set to 3 to use pretrained imagenet
        """

        self.t_pointer = t_pointer
        self.f_pointer = f_pointer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backbone = backbone
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.augment = augment
        self.aug_len = aug_len
        # TODO: get window size from time length
        self.window_size = window_size

    def __get_data(self, index, augment=False):

        time_batch = self.t_pointer[index]
        date = time_batch.strftime("%Y%m%d")
        freq_batch = self.f_pointer[index]

        datadir = list(root_data_dir.glob('SUN_TRACKING_'+date+'*'))[0]
        # a_time = time.time()
        d_ind = np_files.index(datadir)
        tslice = np.searchsorted(ltime[d_ind], time_batch.isot, side='right')
        fslice = np.searchsorted(lfreq[d_ind], freq_batch, side='right')
        # print(tslice, len(ltime[d_ind]), fslice, len(lfreq[d_ind]))
        if tslice > len(ltime[d_ind]) - self.window_size:
            return np.zeros((self.window_size, self.window_size, self.n_channels)), np.zeros((self.window_size, self.window_size, self.num_classes))
        if tslice == 0:
            return np.zeros((self.window_size, self.window_size, self.n_channels)), np.zeros((self.window_size, self.window_size, self.num_classes))
        if fslice > len(lfreq[d_ind]) - self.window_size:
            return np.zeros((self.window_size, self.window_size, self.n_channels)), np.zeros((self.window_size, self.window_size, self.num_classes))
        if fslice == 0:
            return np.zeros((self.window_size, self.window_size, self.n_channels)), np.zeros((self.window_size, self.window_size, self.num_classes))

             
        # print(index, time_batch, freq_batch, datadir)
        # print(ltime[d_ind][tslice])
        # data for 1995 is fake so there's no mask offset
        # CUSUM slope method shifts by 2 
        if time_batch.strftime("%Y") =="1995":
            mask_offset = 0
        else:
            mask_offset = 2 
        if augment:
            # print("shifting window")
            rng = np.random.default_rng()
            # Don't shift time incase train and val sets overlap
            shift = rng.integers(-self.window_size//2, self.window_size//2)

            if (tslice+shift + self.window_size < len(ltime[d_ind]) 
            and tslice+shift > mask_offset):
                # print("old tslice", tslice)
                tslice += shift
            # print("new tslice", tslice)
            shift = rng.integers(-self.window_size//2, self.window_size//2)
            # print("F shift", shift)
            if (fslice+shift + self.window_size < len(lfreq[d_ind]) 
            and fslice+shift> 0):
                # print("old fslice", fslice)
                fslice += shift
            # print("new fslice", fslice)
        # "Normal"
        X = ldata[d_ind][tslice:tslice+self.window_size, fslice:fslice+self.window_size, :self.num_classes]
        y = lmask[d_ind][tslice-mask_offset:tslice-mask_offset+self.window_size, fslice:fslice+self.window_size, :self.num_classes]
        
        # SPIKES
        # X = ldata[d_ind][tslice:tslice+self.window_size, fslice:fslice+self.window_size, :self.n_channels]
        # y = lmask[d_ind][tslice-mask_offset:tslice-mask_offset+self.window_size, fslice:fslice+self.window_size, -1][:,:,np.newaxis]

        # y = np.zeros((self.window_size, self.window_size, 2))
        # y[:,:,0] = oldmask[d_ind][tslice-mask_offset:tslice-mask_offset+self.window_size, fslice:fslice+self.window_size,0]
        # y[:,:,1] = lmask[d_ind][tslice-mask_offset:tslice-mask_offset+self.window_size, fslice:fslice+self.window_size,1]

        # print(X.shape, ldata[d_ind][tslice:tslice+self.window_size, -self.window_size:].shape)
        # if X.shape != (self.window_size, self.window_size)
        # print(y.shape)
        time_layer = Time(ltime[d_ind][tslice:tslice+self.window_size]).mjd
        time_layer = np.tile(time_layer,
                             (self.window_size,1)).T[:, :, np.newaxis]
        # dt_layer = (Time(ltime[d_ind][tslice:tslice+self.window_size]) - Time(ltime[d_ind][tslice])).sec
        # dt_layer = np.tile(dt_layer,
        #                    (self.window_size, 1)).T[:, :, np.newaxis]
        # convert to MHz
        freq_layer = lfreq[d_ind][fslice:fslice+self.window_size]/1e6
        freq_layer = np.tile(freq_layer,
                             (self.window_size, 1))[:, :, np.newaxis]
        X = tf.concat((X,freq_layer), axis=-1)
        # X = tf.concat((X,dt_layer), axis=-1)
        X = tf.concat((X,time_layer), axis=-1)
        X = self.preprocess_input(X)

        return X, y

    def preprocess_input(self,X):
        # TODO: make this work properly
        preprocess_input = sm.get_preprocessing(self.backbone)
        if self.backbone == "vgg16":
            # vgg16 preprocess needs pixel range 0-255
            # this isn't correct so don't us vgg16
            X = X/tf.reduce_max(X)
            X = X*255
            print("ERROR: backbone vgg16 not working properly")
            exit() 
        return preprocess_input(X)

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(seed=42)
            reidx = rng.permutation(self.__len__())
            self.t_pointer = self.t_pointer[reidx]
            self.f_pointer = self.f_pointer[reidx]
        else:
            pass
        
    def __getitem__(self, index, augment=False):
        #index is batch number
        # a_time = time.time()
        X, y = self.__get_data(index, augment=augment)
        # X, y = self.__get_from_index(index)
        # b_time = time.time() - a_time
        # print("time to get batch {}: {}".format(index, b_time))
        return X, y
    def __plotitem__(self, index, dsetname):
        X, y = self.__getitem__(index)
        tlims = [self.t_pointer[index].plot_date, 
                 (self.t_pointer[index] + self.window_size*0.25*u.s).plot_date]
        flims = np.array([self.f_pointer[index],
                 self.f_pointer[index] + self.window_size*2*12207.03125])/1e6
        fig, ax = plt.subplots(2,2, figsize=(7,7), sharex=True, sharey=True)
        stokesI = ax[0,0].imshow(tf.transpose(X[:,:,0]),
                                 aspect='auto',
                                 origin='lower',
                                 cmap='viridis',
                                 extent=[tlims[0],
                                         tlims[1],
                                         flims[0],
                                         flims[1]],
                                 vmin=np.percentile(X[:,:,0], 5),
                                 vmax=np.percentile(X[:,:,0], 95))
        stokesV = ax[1,0].imshow(tf.transpose(X[:,:,1]),
                                 aspect='auto',
                                 origin='lower',
                                 cmap='magma',
                                 extent=[tlims[0],
                                         tlims[1],
                                         flims[0],
                                         flims[1]],
                                 vmin=np.percentile(X[:,:,1], 5),
                                 vmax=np.percentile(X[:,:,1], 95))
        maskI = ax[0,1].imshow(tf.transpose(y[:,:,0]),
                               aspect='auto',
                               origin='lower',
                               cmap='gray',
                               extent=[tlims[0],
                                       tlims[1],
                                       flims[0],
                                       flims[1]],
                               vmin=0,
                               vmax=1)
        maskV = ax[1,1].imshow(tf.transpose(y[:,:,1]),
                               aspect='auto',
                               origin='lower',
                               cmap='gray',
                               extent=[tlims[0],
                                       tlims[1],
                                       flims[0],
                                       flims[1]],
                               vmin=0,
                               vmax=1)

        ax[1,0].xaxis_date()
        
        ax[1,0].xaxis.set_major_formatter(date_format)
        ax[1,0].tick_params(axis='x', labelrotation=45)
        ax[1,1].tick_params(axis='x', labelrotation=45)
        fig.colorbar(stokesI, ax = ax[0,0])
        fig.colorbar(stokesV, ax = ax[1,0])
        fig.colorbar(maskI, ax = ax[0,1])
        fig.colorbar(maskV, ax = ax[1,1])
        fig.suptitle("Start Time: {} Start Freq: {}MHz".format(self.t_pointer[index].isot, np.round(self.f_pointer[index]/1e6,2)))
        fig.supxlabel("Time (UTC)")
        fig.supylabel("Frequency (MHz)")       
        plt.tight_layout()
        plt.savefig("/data/pmurphy/ml_imgs/{}/input_index{}.png".format(dsetname, str(index).zfill(6)))
        fig.clf()
        plt.close("all")
        del fig, ax, X, y, stokesI, stokesV, maskI, maskV
        gc.collect()
    def __call__(self):

        for i in range(len(self.t_pointer)):
            yield self.__getitem__(i)
            if self.augment:
                for a in range(self.aug_len):
                    yield self.__getitem__(i, augment=True)
            if i == len(self.t_pointer)-1:
                self.on_epoch_end()

    def __len__(self):
        if not self.augment:
            return len(self.t_pointer)  # // self.batch_size
        else:
            return (self.aug_len+1) * len(self.t_pointer)

class drop_channel(layers.Layer):
    # remove channel before it goes into the network
    def __init__(self, last_chan):
        super(drop_channel, self).__init__()
        self.last_chan = last_chan
        self.build((None, 256, 256, 4))
    def __call__(self, inputs):
        return inputs[:,:,:,:self.last_chan]

class normalise_time(layers.Layer):
    def __init__(self, time_chan):
        super(normalise_time, self).__init__()
        self.time_chan = time_chan
    
    def __call__(self,  inputs):
        norm_time = inputs[:,:,:,self.time_chan] - inputs[:,0,0,self.time_chan][:, np.newaxis, np.newaxis]
        # new_inputs = tf.concat([inputs[:,:,:,:self.time_chan], norm_time[:,:,:,np.newaxis]],-1)
        return norm_time[:,:,0]

def configure_for_performance(ds):
    # ds = ds.cache()
    ds = ds.shuffle(buffer_size=len(ds), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    # ds = ds.map(drop_channel, num_parallel_calls=AUTOTUNE)
    return ds

# def drop_channel(inp, out):
#     return inp[:,:,:,:3], out

date_format = dates.DateFormatter("%H:%M:%S")


# old_data_dir = pathlib.Path("/minerva/pmurphy/old/")
# remove_date = old_data_dir/"SUN_TRACKING_20220519_101036_0"
# np_files = list(old_data_dir.glob('SUN_TRACKING*'))
# np_files.remove(remove_date)
# np_files.sort()
# oldmask = load_all_files('mask.npy')


root_data_dir = pathlib.Path("/minerva/pmurphy/")
tf.keras.backend.clear_session()
num_classes = 1

np_files = list(root_data_dir.glob('SUN_TRACKING*'))
# if __name__ == "__main__":
#     remove_date = root_data_dir/"SUN_TRACKING_20220519_101036_0"
#     np_files.remove(remove_date)
np_files.sort()
ltime = load_all_files('time.npy')

# ltime = np.load("/data/pmurphy/overlap_times.npy", allow_pickle=True)
lfreq = load_all_files('frequency.npy')
lmask = load_all_files('mask.npy')
# ldata = load_all_files('data_rescale.npy')
ldata = load_all_files('data.npy')
batch_size = 32

# print(lmask[0].shape, ldata[0].shape)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with given parameters')
    parser.add_argument('network',
                        help='network name either "Unet" or "Linknet" case sensitive',
                        default='Unet')
    parser.add_argument('-e','--epochs', type=int, help='number of epochs default=50', default=50)
    parser.add_argument('-b',
                        '--backbone',
                        help='backbone for network, must be compatibale with `segmentation_models`',
                        default='vgg16')
    parser.add_argument('-p',
                        '--plot',
                        help='only plot data, no training',
                        action='store_true')
    args = parser.parse_args()
    net = args.network
    epochs = args.epochs
    backbone = args.backbone
    plot = args.plot
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE
    window_size = 256
    n_channels = 4
    tf.keras.backend.set_floatx('float64')
    # put data into tensorflow datasets

    # imagenet_preprocess = get_preprocessing(backbone)
    # skip first file (fake data)
    # r_times, r_freqs = tile_data(ltime, lfreq, window_size)
    # (r_times, r_freqs), (r_times_val, r_freqs_val) = stratify_sample(ltime, lfreq, window_size, 0.2)
    # r_times = Time(r_times)
    # r_times_val = Time(r_times_val)

    # Need to do proper stratified sampling
    # rng = np.random.default_rng(seed=42)
    # shuff_ind = rng.permutation(len(r_times))
    # r_times = r_times[shuff_ind]
    # r_freqs = r_freqs[shuff_ind]
    # train_val_ratio = 0.7

    # r_times_val = r_times[int(len(r_times)*train_val_ratio):]
    # r_freqs_val = r_freqs[int(len(r_freqs)*train_val_ratio):]

    # r_times_test = r_times_val[len(r_times_val)//2:]
    # r_freqs_test = r_freqs_val[len(r_freqs_val)//2:]

    # r_times_val = r_times_val[:len(r_times_val)//2]
    # r_freqs_val = r_freqs_val[:len(r_freqs_val)//2]

    # r_times = r_times[:int(len(r_times)*train_val_ratio)]
    # r_freqs = r_freqs[:int(len(r_freqs)*train_val_ratio)]

    # add in fake data to train set
    # f_times, f_freqs = tile_data(ltime[0], lfreq[0], window_size)
    # f_times = Time(f_times)
    # r_times = np.concatenate((r_times, f_times))
    # r_freqs = np.concatenate((r_freqs, f_freqs))
    # shuff_ind1 = rng.permutation(len(r_times))
    # r_times = r_times[shuff_ind1]
    # r_freqs = r_freqs[shuff_ind1]


    # print("batch_size {}".format(len(r_times)//len(np_files)))

    # train_gen = DataGen(r_times,
    #                     r_freqs,
    #                     batch_size,
    #                     shuffle=False,
    #                     window_size=window_size,
    #                     backbone=backbone,
    #                     n_channels=n_channels,
    #                     augment=False,)
    #                     # aug_len=5)

    # val_gen = DataGen(r_times_val,
    #                 r_freqs_val,
    #                 batch_size,
    #                 shuffle=False,
    #                 window_size=window_size,
    #                 n_channels=n_channels,
    #                 backbone=backbone)

    # test_gen = DataGen(r_times_test,
    #                 r_freqs_test,
    #                 batch_size,
    #                 shuffle=False,
    #                 window_size=window_size,
    #                 n_channels=n_channels,
    #                 backbone=backbone)

    # train_ds = tf.data.Dataset.from_generator(train_gen,
    #                                         output_signature=(
    #                                         tf.TensorSpec(shape=(window_size,window_size,n_channels),
    #                                                         dtype=tf.float64),
    #                                         tf.TensorSpec(shape=(window_size,window_size,num_classes),
    #                                                         dtype=tf.float64)))

    # val_ds = tf.data.Dataset.from_generator(val_gen,
    #                                         output_signature=(
    #                                         tf.TensorSpec(shape=(window_size,window_size,n_channels),
    #                                                     dtype=tf.float64),
    #                                         tf.TensorSpec(shape=(window_size,window_size,num_classes),
    #                                                     dtype=tf.float64)))

    # test_ds = tf.data.Dataset.from_generator(test_gen,
    #                                         output_signature=(
    #                                         tf.TensorSpec(shape=(window_size,window_size,n_channels),
    #                                                     dtype=tf.float64),
    #                                         tf.TensorSpec(shape=(window_size,window_size,num_classes),
    #                                                     dtype=tf.float64)))

    # train_ds = ds.take(int(len(ds)*train_val_ratio))
    # val_ds = ds.skip(int(len(ds)*train_val_ratio))
    # test_ds = val_ds.take(len(val_ds)//2)
    # val_ds = val_ds.skip(len(val_ds)//2)

    

    # train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(train_gen.__len__()))
    # val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(val_gen.__len__()))
    # test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(test_gen.__len__()))

    # train_ds = configure_for_performance(train_ds)
    # val_ds = configure_for_performance(val_ds)
    # test_ds = configure_for_performance(test_ds)
    n_splits = 5
    test_size = 0.2
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    k = 0 
    if k == 0:
    # for k in range(n_splits):

        train_ds, val_ds = create_train_test_val_ds(ltime, lfreq, sss, window_size, k, spikes=True)
        model_name = '/data/pmurphy/{}_{}_{}epochs_bce_spikes_256_taugment25'.format(net, backbone, epochs)
        print(model_name)
        # for i in range(train_gen.__len__()):
        #     print(i,train_gen.__getitem__(i)[1].shape)
        # # train_gen.__getitem__(1)

        # with strategy.scope():
        if net == 'Unet':
            model = Unet(backbone_name=backbone,
                        # input_shape=(None, None, n_channels),
                        classes=num_classes,
                        encoder_weights='imagenet',
                        encoder_freeze=True,)
                        # decoder_filters=[128, 64, 32, 16, 8])
                        #input_shape=(None, None, 1))
        elif net == 'Linknet':
        # model = Linknet(backbone_name=backbone,
        #                 encoder_weights='imagenet',
        #                 input_shape=(None, None, 1))s
            model = Linknet(backbone_name=backbone,
                        # input_shape=(None, None, n_channels),
                        classes=num_classes,
                        encoder_weights='imagenet',
                        encoder_freeze=True,)
                        #decoder_filters=[1024,512,256,128,64])
        elif net == 'FPN':
            model = FPN(backbone_name=backbone,
                        input_shape=(None, None, n_channels),
                        classes=num_classes,
                        encoder_weights='imagenet',
                        encoder_freeze=True,)
        elif net == 'PSPNet':
            model = PSPNet(backbone_name=backbone,
                        input_shape=(window_size, window_size, n_channels),
                        classes=num_classes,
                        encoder_weights='imagenet',
                        encoder_freeze=True,)
        else:
            print("Invalid network name")





        metric = FScore() #IOUScore()#tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])#tf.keras.metrics.MeanIoU(num_classes=2)
        metric_list = [metric,
                       IOUScore(threshold=0.5),]
                    #    IOUScore(class_indexes=0, name="IOU_I", threshold=0.5),
                    #    IOUScore(class_indexes=1, name="IOU_V", threshold=0.5),]
                    #    IOUScore(class_indexes=2, name="IOU_typeII", threshold=0.5, class_weights=43.),
                    #    IOUScore(class_indexes=3, name="IOU_typeIII", threshold=0.5, class_weights=1.19),
                    #    IOUScore(class_indexes=4, name="IOU_typeIV", threshold=0.5, class_weights=78.8),
                    #    IOUScore(class_indexes=5, name="IOU_ctm", threshold=0.5, class_weights=8.88)]
        weight_list = [1,1] #[1,1,43,1.19,78.8,8.88]
        # loss = sm.losses.bce_jaccard_loss
        loss = BinaryCELoss() #CategoricalCELoss(class_weights=np.array((1,1))) #sm.losses.cce_jaccard_loss #
        lrate0 = 1e-2
        decay_steps = 10 * len(train_ds)
        decay_rate = 0.5
        alpha = 1e-2
        # lrate = CosineDecay(lrate0, 0.2*epochs*len(train_ds), alpha)
        # lrate = CosineDecay(lrate0, 50*len(train_ds), alpha)
        lrate = CosineDecayRestarts(lrate0, 25*len(train_ds), t_mul=1.0, m_mul=1.0, alpha=alpha)
        # lrate = ExponentialDecay(lrate0, decay_steps, decay_rate)
        optimizer= tf.keras.optimizers.Adam(learning_rate=lrate) #tf.keras.optimizers.SGD(momentum=0.99)  #"adam"
        # # Force input to have 3 channels
        inp = layers.Input(shape=(None, None, n_channels))
        ld = drop_channel(3)(inp) # drop last channel
        # l1 = layers.Conv2D(3, (1, 1))(ld) # map N channels data to 3 channels
        
        out = model(ld)
        model = models.Model(inp, out, name=model.name)

        # add regulization to trainable layers
        trainable_layers = np.arange(len(model.layers))[np.array([layer.trainable for layer in model.layers])]
        kernel_regularizer = None #tf.keras.regularizers.L2()
        for layer in np.array(model.layers)[trainable_layers]:
            if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = kernel_regularizer

        if plot:
            custom_objects = {"f1-score":metric_list[0],
                              "iou_score":metric_list[1],
                              "IOU_I":metric_list[2],
                              "IOU_V":metric_list[3],
                              "IOU_typeII":metric_list[4],
                              "IOU_typeIII":metric_list[5],
                              "IOU_typeIV":metric_list[6],
                              "IOU_ctm":metric_list[7],
                              "binary_crossentropy_loss":loss}
            model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=metric_list,
                    loss_weights=weight_list)


        # model.summary()
        checkpoint_path = model_name+"_checkpoints"
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=model_name+'_tboard'),
                                                    #  histogram_freq=1,
                                                    # update_freq='batch'),
                    #  tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=5),
                    #  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                    #                                     save_freq=50*batch_size)
                    ]

        if not plot:
            #train with frozen layers
            #train longer on first run through
            history = model.fit(train_ds,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(val_ds),
                    callbacks=callbacks,
                    )

            # #unfreeze encoder layers n_unfreeze at a time
            unet_layers = model.layers[2].layers
            nontrainable_layers = np.arange(len(unet_layers))[np.array([not layer.trainable for layer in unet_layers])]
            n_unfreeze = len(nontrainable_layers)#//10
            print("number of non-trainable layers: {}".format(len(nontrainable_layers)))
            fine_tune_epochs = 10#epochs//2
            total_epochs = history.epoch[-1]+fine_tune_epochs

            # while len(nontrainable_layers) > 0:
            for layer in np.array(unet_layers)[nontrainable_layers[-n_unfreeze:]]:
                layer.trainable = True
            nontrainable_layers = np.arange(len(unet_layers))[np.array([not layer.trainable for layer in unet_layers])]
            #     #set small learning rate
            print("Fine tuning")
            # total_epochs = history.epoch[-1]+fine_tune_epochs
            lrate1 = lrate(len(train_ds)*history.epoch[-1])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate1/10),
                            loss=loss,
                            metrics=metric_list,
                            loss_weights=weight_list)

        #     print("number of non-trainable layers: {}".format(len(nontrainable_layers)))
            history = model.fit(train_ds,
                epochs=total_epochs,
                initial_epoch=history.epoch[-1],
                verbose=2,
                validation_data=(val_ds),
                callbacks=callbacks,
                )
            model.save(model_name)

        
        output = model_name+'_results/'
        for ds, dsname in zip([train_ds, val_ds], ['train/','val/']):
            batch_iter = iter(ds)
            for i in range(3):
                image_batch, mask_batch = next(batch_iter)
                plot_results(image_batch, mask_batch, 0, model, output+dsname, i)
                # for l in range(2):
                #     plot_results(image_batch, mask_batch, l, model, output+dsname, i)
