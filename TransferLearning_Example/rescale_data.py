#!/usr/bin/env python

from multiprocessing import Pool

import numpy as np

from pathlib import Path
from skimage import exposure

def rescale_data(data_dir):
    print("Rescaling {}".format(data_dir))
    data = np.load(data_dir/"data.npy")
    data_clip = np.clip(data,
                        np.percentile(data, 1, axis=(0,1)),
                        np.percentile(data, 97.5, axis=(0,1)))
    I_rescale = exposure.rescale_intensity(data[:,:,0],
                                           in_range=(np.min(data_clip[:,:,0]),
                                                     np.max(data_clip[:,:,0])))
    V_rescale = exposure.rescale_intensity(data[:,:,1],
                                           in_range=(np.min(data_clip[:,:,1]),
                                                     np.max(data_clip[:,:,1])))
    data_rescale = np.zeros_like(data)
    data_rescale[:,:,0] = I_rescale
    data_rescale[:,:,1] = V_rescale

    np.save(data_dir/"data_rescale.npy", data_rescale, fix_imports=False)
    return
    
root_data_dir = Path("/minerva/pmurphy")
np_files = sorted(root_data_dir.glob('SUN_TRACKING*'))


with Pool() as pool:
    pool.map(rescale_data, np_files)