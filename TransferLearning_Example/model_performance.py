#!/usr/bin/env python

import os

import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf

from astropy.time import Time
from matplotlib import dates
sm.set_framework('tf.keras')
from segmentation_models.losses import BinaryCELoss
from segmentation_models.metrics import IOUScore, FScore
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from data_gen_test import DataGen

def plot_results(image_batch, mask_batch, model, output_dir, im_index):
    #image_batch, out_batch = next(iter(train_ds))
    #decoded_imgs = conv_ae.predict(image_batch)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df = (200/1024)/8  # MHz
    date_format = dates.DateFormatter("%H:%M:%S")
    fig, axs = plt.subplots(2, 3, figsize=(12, 4), sharex=True)#,
    #                           gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    # fig.subplots_adjust(wspace=0.02)

    #image_batch, out_batch = next(iter(train_ds))
    decoded_imgs = model.predict(image_batch)
    axs[0,0].set_title("Input Layer")
    axs[0,1].set_title("Ground Truth Mask")
    axs[0,2].set_title("Predicted Mask")
    for layer in [0,1]:
        im_vmin = np.percentile(image_batch[0][:,:,layer],5)
        im_vmax = np.percentile(image_batch[0][:,:,layer],95)
        farr = df*np.arange(image_batch[0][:,:,0].shape[0]) + image_batch[0][0,0,2]
        start_time = Time(image_batch[0][0,0,3], format='mjd')
        tarr = 0.25*np.arange(image_batch[0][:,:,0].shape[1])*u.s + start_time
        cmap_list=['viridis', 'inferno']
        real_im = axs[layer,0].imshow(tf.transpose(image_batch[0][:,:,layer]),
                                    cmap=cmap_list[layer],
                                    aspect='auto',
                                    origin='lower',
                                    extent=[tarr[0].plot_date, tarr[-1].plot_date, farr[0], farr[-1]],
                                    vmin=im_vmin,
                                    vmax=im_vmax)
        real_mask = axs[layer,1].imshow(tf.transpose(mask_batch[0][:,:,layer]),
                                    cmap=cmap_list[layer],
                                    aspect='auto',
                                    origin='lower',
                                    extent=[tarr[0].plot_date, tarr[-1].plot_date, farr[0], farr[-1]],
                                    vmin=0,
                                    vmax=1)
        pred_mask = axs[layer,2].imshow(tf.transpose(decoded_imgs[0][:,:,layer]),
                                    cmap=cmap_list[layer],
                                    aspect='auto',
                                    origin='lower',
                                    extent=[tarr[0].plot_date, tarr[-1].plot_date, farr[0], farr[-1]],
                                    vmin=0,
                                    vmax=1)
                                    #     divider = make_axes_locatable(axs[layer,i,2])
                                    #     cax1 = divider.append_axes("right", size="5%", pad=0.05)
                                    #     cax1 = fig.add_axes([0, 0, 0.1, 1])
        fig.colorbar(real_im, ax=axs[layer,0])
        fig.colorbar(real_mask, ax=axs[layer,1])
        fig.colorbar(pred_mask, ax=axs[layer,2])
        if np.max(decoded_imgs[0][:,:,layer]) >= 0.8:
            T, F = np.meshgrid(tarr.plot_date, farr)
            axs[layer,0].contour(T, F, tf.transpose(decoded_imgs[0][:,:,layer]), [0.8], colors='white')
    
    # axs[0].set_title("Start Time {}".format(start_time.isot))
    axs[0,2].xaxis.set_major_formatter(date_format)
    # axs[2].tick_params(axis='x', rotation=45)
    # axs[i,0].axis("off")
    # axs[i,1].axis("off")
    # axs[i,2].axis("off")
    fig.supylabel("Frequency (MHz)")
    fig.supxlabel("Time on {}".format(start_time.isot[:10]))

    outfile = output_dir+"compare_{}.png".format(str(im_index).zfill(5))
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close('all')

def concat_batches(image_batch, mask_batch, model):
    X_c = tf.concat([*image_batch], axis=1)
    y_c = tf.concat([*mask_batch], axis=1)
    y_pred= model.predict(image_batch)
    y_cpred = tf.concat([*y_pred], axis=1)
    return X_c, y_c, y_cpred

def percentage_masked(image_batch, mask_batch, model, axis=(0,1)):
    _,_, y_cpred = concat_batches(image_batch, mask_batch, model)
    if axis == (0,1):
        percentage_masked = np.sum(y_cpred, axis=axis)/(y_cpred.shape[0]*y_cpred.shape[1])
    elif axis == 1:
        percentage_masked = np.sum(y_cpred, axis=axis)/y_cpred.shape[0]

    elif axis == 0:
        percentage_masked = np.sum(y_cpred, axis=axis)/y_cpred.shape[1]

    else:
        print("axis error")
    return percentage_masked



def percentage_masked_relative(image_batch, mask_batch, model, class0, class1):
    _,_, y_cpred = concat_batches(image_batch, mask_batch, model)
    percentage_masked = np.sum(y_cpred, axis=(0,1))[class1]/np.sum(y_cpred, axis=(0,1))[class0]
    return percentage_masked

def plot_full_freqband(image_batch, mask_batch, model, layer, output_dir, im_index):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    X_c, y_c, y_cpred = concat_batches(image_batch, mask_batch, model)
    df = (200/1024)/8  # MHz
    date_format = dates.DateFormatter("%H:%M:%S")
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].set_title("Input Layer")
    axs[1].set_title("Ground Truth Mask")
    axs[2].set_title("Predicted Mask")
    # for layer in [0,1]:
    im_vmin = 0#np.percentile(X_c[:,:,layer],5)
    im_vmax = 0.25#np.percentile(X_c[:,:,layer],95)
    farr = df*np.arange(X_c[:,:,0].shape[1]) + X_c[0,0,2]
    
    start_time = Time(X_c[0,0,3], format='mjd')
    tarr = 0.25*np.arange(X_c[:,:,0].shape[0])*u.s + start_time
    
    cmap_list = ['viridis', 'inferno', 'Purples_r', 'Oranges_r', 'Greens_r','Blues_r']
    real_im = axs[0].imshow(tf.transpose(X_c[:,:,0]),
                                cmap=cmap_list[0],
                                aspect='auto',
                                origin='lower',
                                extent=[tarr[0].plot_date, tarr[-1].plot_date, farr[0], farr[-1]],
                                vmin=im_vmin,
                                vmax=im_vmax)
    real_mask = axs[1].imshow(tf.transpose(y_c[:,:,layer]),
                                cmap=cmap_list[layer],
                                aspect='auto',
                                origin='lower',
                                extent=[tarr[0].plot_date, tarr[-1].plot_date, farr[0], farr[-1]],
                                vmin=0,
                                vmax=1)
    pred_mask = axs[2].imshow(tf.transpose(y_cpred[:,:,layer]),
                                cmap=cmap_list[layer],
                                aspect='auto',
                                origin='lower',
                                extent=[tarr[0].plot_date, tarr[-1].plot_date, farr[0], farr[-1]],
                                vmin=0,
                                vmax=1)

    fig.colorbar(real_im, ax=axs[0])
    fig.colorbar(real_mask, ax=axs[1])
    fig.colorbar(pred_mask, ax=axs[2])
    if np.max(y_cpred[:,:,layer]) >= 0.8:
        T, F = np.meshgrid(tarr.plot_date, farr)
        
        axs[0].contour(T, F, tf.transpose(y_cpred[:,:,layer]), [0.8], colors='white')
    
    # axs[0].set_title("Start Time {}".format(start_time.isot))
    axs[2].xaxis.set_major_formatter(date_format)
    # axs[2].tick_params(axis='x', rotation=45)
    # axs[i,0].axis("off")
    # axs[i,1].axis("off")
    # axs[i,2].axis("off")
    fig.supylabel("Frequency (MHz)")
    fig.supxlabel("Time on {}".format(start_time.isot[:10]))

    outfile = output_dir+"layer_{}_compare_{}.png".format(layer, str(im_index).zfill(5))
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close('all')


def configure_for_performance(ds):
    ds = ds.cache()
    # ds = ds.shuffle(buffer_size=len(ds), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # ds = ds.map(drop_channel, num_parallel_calls=AUTOTUNE)
    return ds

AUTOTUNE = tf.data.AUTOTUNE
window_size = 256
backbone = "resnet34"
n_channels = 4 
num_classes = 2
tf.keras.backend.set_floatx('float64')
date_format = dates.DateFormatter("%H:%M:%S")
time = np.load("/minerva/pmurphy/SUN_TRACKING_20220519_101036_0/time.npy")
freq = np.load("/minerva/pmurphy/SUN_TRACKING_20220519_101036_0/frequency.npy")

times = time[2:len(time)-window_size-2][::40] #  image every X seconds
freqs = np.tile(freq[::256][:-1], (len(times))) # freq[::window_size] * np.ones(len(times)) #somewhere in the middle

# print(times)
# print(freqs)
times = Time(np.tile(times, (len(freq[::256][:-1]),1)).T.ravel())
batch_size = len(freq[::256][:-1])

ds_gen = DataGen(times,
                 freqs,
                 batch_size,
                 shuffle=False,
                 window_size=window_size,
                 backbone=backbone,
                 n_channels=n_channels,
                 augment=False)

ds = tf.data.Dataset.from_generator(ds_gen,
                                    output_signature=(
                                    tf.TensorSpec(shape=(window_size,window_size,n_channels),
                                                    dtype=tf.float64),
                                    tf.TensorSpec(shape=(window_size,window_size,num_classes),
                                                    dtype=tf.float64)))
ds = ds.apply(tf.data.experimental.assert_cardinality(ds_gen.__len__()))
ds = configure_for_performance(ds)

metric = FScore()
metric_list = [metric,
                IOUScore(threshold=0.5),
                IOUScore(class_indexes=0, name="IOU_I", threshold=0.5),
                IOUScore(class_indexes=1, name="IOU_V", threshold=0.5),]
                # IOUScore(class_indexes=2, name="IOU_typeII", threshold=0.5, class_weights=43.),
                # IOUScore(class_indexes=3, name="IOU_typeIII", threshold=0.5, class_weights=1.19),
                # IOUScore(class_indexes=4, name="IOU_typeIV", threshold=0.5, class_weights=78.8),
                # IOUScore(class_indexes=5, name="IOU_ctm", threshold=0.5, class_weights=8.88)]

model_name = '/data/pmurphy/Unet_resnet152_120epochs_bce_2classes_augment_largercosinedecay_kfold0'
# loss = sm.losses.bce_jaccard_loss
loss = BinaryCELoss()
lrate0 = 1e-2
decay_steps = 10 * len(ds)
decay_rate = 0.5
lrate = ExponentialDecay(lrate0, decay_steps, decay_rate)
optimizer= tf.keras.optimizers.Adam()#(learning_rate=lrate)

custom_objects = {"f1-score":metric_list[0],
                    "iou_score":metric_list[1],
                    "IOU_I":metric_list[2],
                    "IOU_V":metric_list[3],
                    # "IOU_typeII":metric_list[4],
                    # "IOU_typeIII":metric_list[5],
                    # "IOU_typeIV":metric_list[6],
                    # "IOU_ctm":metric_list[7],
                    "binary_crossentropy_loss":loss}
model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
model.compile(optimizer=optimizer,
        loss=loss,
        metrics=metric_list)

# custom_objects = {"f1-score":metric_list[0], 
#                 "IOU_I":metric_list[1],
#                 "IOU_V":metric_list[2],
#                 "binary_crossentropy_loss":loss}

# model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)

# model.compile(optimizer=optimizer,
#             loss=loss,
#             metrics=metric_list)

output = model_name+'_sequential/'
iterds = iter(ds)
ious = np.zeros((num_classes, len(ds)))
p_masks = np.zeros((num_classes, len(ds)))
p_masks_t = []
p_masks_f = []
p_mask_rels = np.zeros(len(ds))
for i in range(len(ds)):
    X, y = next(iterds)

    # plot_results(X, y, model, output, i)
    # for layer in range(6):
    #     plot_full_freqband(X, y, model, layer, output, i)
    IOU_I = metric_list[1]
    IOU_V = metric_list[2]

    # decoded_imgs = model.predict(X)
    # ypred = tf.cast(decoded_imgs, tf.float64)
    # iou_i = IOU_I(y, ypred).numpy()
    # iou_v = IOU_V(y, ypred).numpy()
    # ious[0,i] = iou_i
    # ious[1,i] = iou_vs

    p_mask = percentage_masked(X, y, model)
    p_mask_t = percentage_masked(X, y, model, axis=0)
    p_mask_f = percentage_masked(X, y, model, axis=1)
    # p_mask_rel = percentage_masked_relative(X, y, model, 0, 1)
    # p_mask_rels[i] = p_mask_rel
    p_masks[:,i] = p_mask
    p_masks_t.append(p_mask_t)
    p_masks_f.append(p_mask_f)

p_masks_t = np.array(p_masks_t)
p_masks_f = np.array(p_masks_f)

fig, ax = plt.subplots(2,1)
ax[0].imshow(p_masks_t[:,:,0])
ax[1].imshow(p_masks_t[:,:,1])
plt.savefig(output+"percent_mask_t.png")

fig, ax = plt.subplots(2,1)
ax[0].imshow(p_masks_f[:,:,0])
ax[1].imshow(p_masks_f[:,:,1])
plt.savefig(output+"percent_mask_f.png")
# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(times[::batch_size].plot_date, ious[0], '-', label='I')
# ax.plot(times[::batch_size].plot_date, ious[1], '--', label='V')
# ax.set_ylabel('IOU')
# ax.set_xlabel('Time')
# ax.xaxis.set_major_formatter(date_format)
# plt.legend()
# plt.savefig(output+"iou_compare.png")

# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(times[::batch_size].plot_date, p_masks[0], '-', label='I')
# ax.plot(times[::batch_size].plot_date, p_masks[1], '--', label='V')
# ax.set_ylabel('Percentage Masked')
# ax.set_xlabel('Time')
# ax.xaxis.set_major_formatter(date_format)
# plt.legend()
# plt.savefig(output+"Percentage_Masked_compare.png")

# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(times[::batch_size].plot_date, p_mask_rels, '-')
# ax.set_ylabel('Percentage Masked Relative')
# ax.set_xlabel('Time')
# ax.xaxis.set_major_formatter(date_format)
# # plt.legend()
# plt.savefig(output+"Percentage_Masked_Relative_compare.png")

# for i, iou in zip(['I', 'V'], ious):
#     # iou_hist, bins = np.histogram(iou, bins=np.arange(10)/10)
#     plt.figure()
#     plt.hist(iou, bins=np.arange(10)/10)
#     mean_iou = np.mean(iou)
#     plt.axvline(mean_iou, c='r', label="Mean: {}".format(np.round(mean_iou, 2)))
#     plt.xlabel("IOU "+i)
#     plt.legend()
#     plt.savefig(output+"IOU_"+i+"_hist.png")
