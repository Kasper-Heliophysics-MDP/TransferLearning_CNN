#!/usr/bin/env python

import os

import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import segmentation_models as sm
import tensorflow as tf

from astropy.time import Time
from matplotlib import dates
sm.set_framework('tf.keras')
from segmentation_models.losses import BinaryCELoss
from segmentation_models.metrics import IOUScore, FScore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit

from data_gen_test import load_all_files, create_train_test_val_ds

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=len(ds), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # ds = ds.map(drop_channel, num_parallel_calls=AUTOTUNE)
    return ds

model_name = "/data/pmurphy/Unet_resnet34_60epochs_bce_morelabels_weights_kfold0"

root_data_dir = pathlib.Path("/data/pmurphy/")
tf.keras.backend.clear_session()
num_classes=2

ltime = load_all_files('time.npy')
lfreq = load_all_files('frequency.npy')
lmask = load_all_files('mask.npy')
ldata = load_all_files('data.npy')

backbone = "resnet34"
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
window_size = 256
n_channels = 4

test_size = 0.2
k = 0
n_splits = 5
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
train_ds, val_ds = create_train_test_val_ds(ltime, lfreq, sss, window_size, k,
                                            batch_size=batch_size, backbone='resnet34', n_channels=n_channels)
print("get small val sample")
val_ds = val_ds.take(1000)
# X, y = train_gen.__getitem__(0)
# print(X[:,:,2])
# metric_list = [FScore(),
#                IOUScore(class_indexes=0, name="IOU_I"),
#                IOUScore(class_indexes=1, name="IOU_V")]
loss = BinaryCELoss()#sm.losses.bce_jaccard_loss
metric = FScore()
metric_list = [metric,
                IOUScore(threshold=0.5),
                IOUScore(class_indexes=0, name="IOU_I", threshold=0.5),
                IOUScore(class_indexes=1, name="IOU_V", threshold=0.5),
                IOUScore(class_indexes=2, name="IOU_typeII", threshold=0.5, class_weights=43.),
                IOUScore(class_indexes=3, name="IOU_typeIII", threshold=0.5, class_weights=1.19),
                IOUScore(class_indexes=4, name="IOU_typeIV", threshold=0.5, class_weights=78.8),
                IOUScore(class_indexes=5, name="IOU_ctm", threshold=0.5, class_weights=8.88)]
# custom_objects = {"f1-score":metric_list[0], 
#                           "IOU_I":metric_list[1],
#                           "IOU_V":metric_list[2],
#                           "binary_crossentropy_plus_jaccard_loss":loss}
custom_objects = {"f1-score":metric_list[0],
                    "iou_score":metric_list[1],
                    "IOU_I":metric_list[2],
                    "IOU_V":metric_list[3],
                    "IOU_typeII":metric_list[4],
                    "IOU_typeIII":metric_list[5],
                    "IOU_typeIV":metric_list[6],
                    "IOU_ctm":metric_list[7],
                    "binary_crossentropy_loss":loss}
print("load model")
model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
unet_layers = model.layers[-1]
bottle_neck_layer = unet_layers.layers[[layer.name for layer in unet_layers.layers].index('relu1')-1]
bottle_neck_input = tf.keras.Model(model.input, model.layers[1].output)
bottle_neck_output = tf.keras.Model(unet_layers.input, bottle_neck_layer.output)

# print(model.summary())
# print(bottle_neck_input.summary())
# print(bottle_neck_output.summary())
print("pedict bottleneck")
ybn_list = []
for i, b in enumerate(iter(val_ds)):
    print("Predict batch:", i)
    Xbn = bottle_neck_input.predict(b[0])
    ybn = bottle_neck_output.predict(Xbn)

    ybn = ybn.reshape(ybn.shape[0], -1)

    ybn_list.append(ybn)
print("start PCA")
print(ybn.shape)
ybn = np.array(ybn_list[:-1])
ybn = ybn.reshape(-1, ybn.shape[-1])
print(ybn.shape)
pca = PCA(n_components=0.95)
pca_fit = pca.fit_transform(ybn)
n_pca_components = len(pca.explained_variance_ratio_)

print("number of PCA components:", n_pca_components)
# print("explained variance ratio", pca.explained_variance_ratio_)
print("start tSNE")
tsne = TSNE(n_components=2,
           learning_rate='auto',
           init='random')
tsne_fit = tsne.fit_transform(ybn)
print("plotting")
plt.plot(pca_fit[:,0], pca_fit[:,1], 'o')
plt.savefig(model_name+"_PCA.png")

plt.figure()
plt.plot(np.arange(n_pca_components), pca.explained_variance_ratio_, 'o')
plt.savefig(model_name+"_scree_plot.png")

plt.figure()
plt.plot(tsne_fit[:,0], tsne_fit[:,1], 'o')
plt.savefig(model_name+"_TSNE.png")