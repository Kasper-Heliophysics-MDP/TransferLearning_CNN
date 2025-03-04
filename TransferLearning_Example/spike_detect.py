#!/usr/bin/env python
import argparse
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import itertools
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import segmentation_models as sm

from pathlib import Path

from astropy.time import Time
from matplotlib import dates
from matplotlib.patches import Rectangle
from skimage import exposure

from nenupy.undysputed import Dynspec

def load_data_from_ds(
        file_names,
        rebin_dt=0.02097152*u.s,
        rebin_df=((200e6/1024)/32)*u.Hz,
        bp_correction="standard",
        fmin=10*u.MHz,
        fmax=90*u.MHz,
        time_range = None,
        stoke="I"):
    """
    Helper function to return data from UnDySPuTeD files.
    Should work ok for small time/frequency ranges.
    Caution advised for time ranges > ~10 mins
    """
    ds = Dynspec(lanefiles=file_names)
    ds.bp_correction = bp_correction
    ds.dispersion_measure = None
    ds.jump_correction = False

    ds.rebin_dt = rebin_dt
    ds.rebin_df = rebin_df
    print("Loading {}".format(file_names))
    ds.freq_range = [fmin, fmax]
    #chop off first and last second in case obeservations aren't aligned
    if time_range is None:
        ds.time_range = [ds.tmin + 1*u.s, ds.tmax - 1*u.s]
    else:
        ds.time_range = time_range
    result = ds.get(stokes=stoke)

    return result

def drift_rate_line(time_array, drift_rate, f0, t0, index=True):
    #y-y0 = m(x-x0)
    if index:
        drift_rate =  drift_rate//(pix_df/pix_dt)
    if drift_rate != 0:
        
        freq_array = (drift_rate*(time_array - t0)) + f0
    else:
        freq_array = f0*np.ones(len(time_array))
    return freq_array.astype(int)
    

