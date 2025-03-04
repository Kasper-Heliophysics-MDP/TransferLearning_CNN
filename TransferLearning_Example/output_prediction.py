#!/usr/bin/env python

import argparse
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from matplotlib import dates
from pathlib import Path
sm.set_framework('tf.keras')
from segmentation_models.losses import BinaryCELoss
from segmentation_models.metrics import IOUScore, FScore
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from data_gen_test import DataGen

def concat_batches(image_batch, mask_batch, model):
    X_c = tf.concat([*image_batch], axis=1)
    y_c = tf.concat([*mask_batch], axis=1)
    y_pred= model.predict(image_batch)
    y_cpred = tf.concat([*y_pred], axis=1)
    return X_c, y_c, y_cpred

def configure_for_performance(ds):
    ds = ds.cache()
    # ds = ds.shuffle(buffer_size=len(ds), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # ds = ds.map(drop_channel, num_parallel_calls=AUTOTUNE)
    return ds

if __name__ == "__main__":
    print("HI")
    parser = argparse.ArgumentParser(description='Make prediction of a given dynamic spectra.')
    parser.add_argument('input_dir',
                        help='Path to preprocessed dynamic spectrum.')
    parser.add_argument('model',
                        help='path to AI model.')
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    model = Path(args.model)
    print("input dir:", input_dir)
    print("model", model)
    AUTOTUNE = tf.data.AUTOTUNE
    window_size = 256
    backbone = "resnet34"
    n_channels = 4 
    num_classes = 2
    tf.keras.backend.set_floatx('float64')

    time = np.load(input_dir/"time.npy")
    freq = np.load(input_dir/"frequency.npy")

    # Prepare batches as sequential tiles to stitch back together later
    
    times = time[2:len(time)-window_size-2][::window_size]
    freqs = np.tile(freq[::window_size][:-1], (len(times)))

    times = Time(np.tile(times, (len(freq[::window_size][:-1]),1)).T.ravel())
    batch_size = len(freq[::window_size][:-1])
    print("generate batches")
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
                                        tf.TensorSpec(shape=(window_size,
                                                             window_size,
                                                             n_channels),
                                                        dtype=tf.float64),
                                        tf.TensorSpec(shape=(window_size,
                                                             window_size,
                                                             num_classes),
                                                        dtype=tf.float64)))
    ds = ds.apply(tf.data.experimental.assert_cardinality(ds_gen.__len__()))
    ds = configure_for_performance(ds)

    
    metric_list = [FScore(),
                IOUScore(threshold=0.5),
                IOUScore(class_indexes=0, name="IOU_I", threshold=0.5),
                IOUScore(class_indexes=1, name="IOU_V", threshold=0.5),]
                # IOUScore(class_indexes=2, name="IOU_typeII", threshold=0.5, class_weights=43.),
                # IOUScore(class_indexes=3, name="IOU_typeIII", threshold=0.5, class_weights=1.19),
                # IOUScore(class_indexes=4, name="IOU_typeIV", threshold=0.5, class_weights=78.8),
                # IOUScore(class_indexes=5, name="IOU_ctm", threshold=0.5, class_weights=8.88)]
    custom_objects = {"f1-score":metric_list[0],
                    "iou_score":metric_list[1],
                    "IOU_I":metric_list[2],
                    "IOU_V":metric_list[3],
                    # "IOU_typeII":metric_list[4],
                    # "IOU_typeIII":metric_list[5],
                    # "IOU_typeIV":metric_list[6],
                    # "IOU_ctm":metric_list[7],
                    "binary_crossentropy_loss":BinaryCELoss()}
    model = tf.keras.models.load_model(model, custom_objects=custom_objects)
    print("start predictions")
    iterds = iter(ds)
    batch_stitch = []
    time_stitch = []
    freq_stitch = []
    for i in range(len(ds)):
        print("predicing batch {} of {}".format(i, len(ds)))
        X, y = next(iterds)
        X_c, y_c, y_cpred = concat_batches(X, y, model)
        freq_c = X_c[0,:,2]
        time_c = X_c[:,0,3]
        batch_stitch.append(y_cpred)
        time_stitch.append(time_c)
        freq_stitch.append(freq_c)
    print("concatenating")
    batch_stitch = np.concatenate(batch_stitch, axis=0)
    time_stitch = np.concatenate(time_stitch, axis=0)
    freq_stitch = freq_c#np.concatenate(freq_stitch, axis=0)
    # print("plotting")
    # plt.imshow(batch_stitch[:,:,0].T,
    #            aspect="auto",
    #            origin="lower",
    #            extent=[
    #                time_stitch[0],
    #                time_stitch[-1],
    #                freq_stitch[0],
    #                freq_stitch[-1]
    #            ])
    # T, F = np.meshgrid(time_stitch, freq_stitch)
    # cs = plt.contour(T, F, batch_stitch[:,:,0].T, [0.9])
    out_fits = "/minerva/pmurphy/predictions/{}.fits".format(input_dir.stem)
    print("saving")
    # plt.savefig("/data/pmurphy/batch_stitch_test.png")
    # np.save("/data/pmurphy/batch_stitch_test.npy", batch_stitch)
    granule_uid = input_dir.stem
    granule_gid = "SUN_TRACKING" #granule_uid.split('_')[0]
    obs_id = granule_uid[:-2]
    dataproduct_type = "ds"
    measurement_type = (
        "phot.flux.density;",
        "phys.polarization.linear;",
        "em.radio;",
        "meta.modelled"
        )
    measurement_type = "".join(measurement_type)
    processing_level = 5
    target_name = "Sun"
    target_class = "star"
    target_region = "solar-corona"
    time_min = Time(time_stitch[0], format="mjd").jd
    time_max = Time(time_stitch[-1], format="mjd").jd
    time_sampling_step_min = (Time(time_stitch[1], format="mjd")
                              - Time(time_stitch[0], format="mjd")).sec
    time_sampling_step_max =time_sampling_step_min
    time_exp_min = time_sampling_step_min
    time_exp_max = time_sampling_step_max
    spectral_range_min = freq_stitch.numpy()[0]*1e6
    spectral_range_max = freq_stitch.numpy()[-1]*1e6
    spectral_sampling_step_min = (freq_stitch.numpy()[1] - freq_stitch.numpy()[0])*1e6
    spectral_sampling_step_max = spectral_sampling_step_min
    spectral_resolution_min =  spectral_range_min \
            / spectral_sampling_step_max
    spectral_resolution_max = spectral_range_max \
            / spectral_sampling_step_min
    spatial_frame_type = "none"
    instrument_host_name = "simulation"
    instrument_name = "NenuFAR_Solar_UNET_v0.01"
    service_title = "solar_nenufar_ml"
    creation_date = Time.now().iso
    modification_date = Time.now().iso
    release_date = Time.now().iso
    access_url = "tbd"
    access_format = "application/fits"
    # access_estsize = ??
    thumbnail_url = "tbd"
    file_name = out_fits
    publisher = "nancay-radio-observatory"
    observer_institute = "nancay-radio-observatory"
    observer_location = "Nancay Radio Observatory, Nancay, Centre-Val de Loire, France"
    observer_lat = 47.376511
    observer_lon = 2.1924

    EPNCore_dict = {
        "granule_uid": [granule_uid],
        "granule_gid": [granule_gid],
        "obs_id": [obs_id],
        "dataproduct_type": [dataproduct_type],
        "measurement_type": [measurement_type],
        "processing_level": [processing_level],
        "target_name": [target_name],
        "target_class": [target_class],
        "target_region": [target_region],
        "time_min": [time_min],
        "time_max": [time_max],
        "time_sampling_step_min": [time_sampling_step_min],
        "time_sampling_step_max": [time_sampling_step_max],
        "time_exp_min": [time_exp_min],
        "time_exp_max": [time_exp_max],
        "spectral_range_min": [spectral_range_min],
        "spectral_range_max": [spectral_range_max],
        "spectral_sampling_step_min": [spectral_sampling_step_min],
        "spectral_sampling_step_max": [spectral_sampling_step_max],
        "spectral_resolution_min": [spectral_resolution_min],
        "spectral_resolution_max": [spectral_resolution_max],
        "spatial_frame_type": [spatial_frame_type],
        "instrument_host_name": [instrument_host_name],
        "instrument_name": [instrument_name],
        "service_title": [service_title],
        "creation_date": [creation_date],
        "modification_date": [modification_date],
        "release_date": [release_date],
        "access_url": [access_url],
        "access_format": [access_format],
        # "access_estsize": [access_estsize],
        "thumbnail_url": [thumbnail_url],
        "file_name": [file_name],
        "publisher": [publisher],
        "observer_institute": [observer_institute],
        "observer_location": [observer_location],
        "observer_lat": [observer_lat],
        "observer_lon": [observer_lon],
        "time_refposition": ["TOPOCENTER"],
        "time_scale": ["UTC"],
        "receiver_name": ["nancay-radio-telescope#LWA"]

    }

    primary_hdu = fits.PrimaryHDU(batch_stitch)
    header  = primary_hdu.header
    # col_list = []
    # for key in EPNCore_dict.keys():
    #     print(key, EPNCore_dict[key])
    #     header[key] = EPNCore_dict[key]
        # if type(EPNCore_dict[key]) is str:
        #     col_format = "{}A".format(len(EPNCore_dict[key]))
        # else:
        #     col_format = "E"
        # col = fits.Column(name=key, format=col_format, array=EPNCore_dict[key])
        # col_list.append(col)
    epn_table = Table(EPNCore_dict)
    bin_hdu_epn = fits.BinTableHDU(epn_table, name="EPN")    
    # col_t = fits.Column(name="Time", array=time_stitch, format='D')
    # bin_hdu_t = fits.BinTableHDU.from_columns([col_t])  
    # col_f = fits.Column(name="Frequency", array=freq_stitch, format='D')
    # bin_hdu_f = fits.BinTableHDU.from_columns([col_f])  
    image_hdu_t = fits.ImageHDU(time_stitch, name="Time")
    image_hdu_f = fits.ImageHDU(freq_stitch, name="Frequency")
    # image_hdu_cs = fits.ImageHDU(cs.allsegs[0][0], name="Contour Coordinates")
    hdul = fits.HDUList([primary_hdu, bin_hdu_epn, image_hdu_t, image_hdu_f])
    
    hdul.writeto(out_fits, overwrite=True)
    print("BYE")