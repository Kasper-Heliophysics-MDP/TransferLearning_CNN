#!/usr/bin/env python
import argparse
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import itertools
import time

import astropy.units as u
# import holoviews as hv
# hv.extension('bokeh')
# from holoviews import dim
# from holoviews.operation.datashader import rasterize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import segmentation_models as sm

from pathlib import Path

from astropy.time import Time
from matplotlib import dates
from matplotlib.patches import Rectangle
from classification_models.keras import Classifiers
from segmentation_models import Linknet, Unet
from segmentation_models.losses import BinaryCELoss
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
from segmentation_models.models._common_blocks import Conv2dBn
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras import layers, losses, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay, CosineDecayRestarts, LearningRateSchedule

from nenupy.undysputed import Dynspec

class DataGenerator:
    def __init__(self, data_dir, train, blank):
        self.data_dir = data_dir
        self.train = train
        self.blank = blank
        
    def getdata(self, data_file):
        mask_file_name = data_file.name.replace('data','mask')
        mask_file = self.data_dir/mask_file_name
        
        spike_id = int(data_file.name.split("id_")[1][:3])
        augment_id = data_file.name.split("augment_")[1][:3] +".npy"
        mask_file0 = self.data_dir/mask_file.name.replace(augment_id, '000.npy')
        load = np.load(data_file, allow_pickle=True).item()
        data = load['data']
        time_array = load['time']
        freq_array = load['frequency']
        mask = np.load(mask_file, allow_pickle=True)
        mask0 = np.load(mask_file0, allow_pickle=True)
        freq_tile = np.tile(freq_array,
                            (data.shape[0], 1))[:,:,np.newaxis]
        time_tile = np.tile(Time(time_array).plot_date,
                            (data.shape[1], 1)).T[:,:,np.newaxis]
#         if self.train:
#             rng = np.random.default_rng()
#             if rng.random() > 0.5:
#                 noise = rng.normal(0,0.25, (data.shape[0],data.shape[1]))
#                 data[:,:,0] += noise


        pix_df = freq_array[1] - freq_array[0]
        pix_dt = (Time(time_array[1]) - Time(time_array[0])).sec
        if not self.blank:
            burst_params = df.loc[spike_id].drop(
                                [
                                    'time_at_maximum_intensity',
                                    'minimum_frequency',
                                    'maximum_frequency']).to_numpy()
            time_location_pixel = np.argmin(np.abs(Time(time_array) - Time(df['time_at_maximum_intensity'][spike_id])))/len(time_array)
            burst_params = np.append(burst_params, time_location_pixel)
            
            #convert freq values to fraction of pixel space
            # burst_params[2] = burst_params[2]/ (pix_df/pix_dt)
            burst_params[3] = np.argmin(np.abs(burst_params[3] - freq_array))/len(freq_array)
            objectness = np.sum(mask)/np.sum(mask0)
        else:
            burst_params = np.zeros(7)
            objectness = 0.
        # burst_params = np.array((burst_params[3], burst_params[6]))
        X = tf.concat((data, freq_tile), axis=-1)
        X = tf.concat((X, time_tile), axis=-1)
        y = mask
        
        return X, (y, burst_params, objectness)
    
    def __getitem__(self, data_file):
        X, (y, burst_params, objectness) = self.getdata(data_file)
        return X, (y, burst_params, objectness) 
    
    def __call__(self):
        for data_file in self.data_dir.glob("data*010.npy"):
            spike_id = int(data_file.name.split("id_")[1][:3])
            if spike_id in df.index:
                yield self.__getitem__(data_file)
    
    def __len__(self):
        file_counter = 0
        for data_file in self.data_dir.glob("data*010.npy"):
            spike_id = int(data_file.name.split("id_")[1][:3])
            if spike_id in df.index:
                file_counter += 1
        return file_counter
    
def configure_for_performance(ds, batch_size=32, shuffle=True):
    # batch_size = 32
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(ds), seed=42)
    ds = ds.batch(batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

class drop_channel(layers.Layer):
    # remove channel before it goes into the network
    def __init__(self, last_chan):
        super(drop_channel, self).__init__()
        self.last_chan = last_chan
        self.activation = layers.Activation('linear', name='data')
        
        # self.build((None, 256, 256, 4))
    def __call__(self, inputs):
        inputs = self.activation(inputs[:,:,:,:self.last_chan])
        # naming_layer = layers.Lambda(lambda x: x, name='data')
        # inputs = naming_layer(inputs)
        return inputs
    
def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = {
        'backend': tf.keras.backend,
        'models': tf.keras.models,
        'layers': tf.keras.layers,
        'utils': tf.keras.utils,
        # 'kernel_regularizer': tf.keras.regularizers.L2(),
    }

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper

def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper

def histogram_loss(y_true, y_pred):
    print(y_true)
    print(" ")
    print(y_pred)
    
    # (true_mask, true_params, _) = tf.split(y_true, num_or_size_splits=3, axis=1)
    # (pred_mask, pred_params, _) = tf.split(y_pred, num_or_size_splits=3, axis=1)
    true_mask, true_params, _ = y_true
    pred_mask, pred_params, _ = y_pred
    hist_true, bins = np.histogram(true_params[:,i], bins=50)
    hist_pred, _ =  np.histogram(pred_params[:,i], bins=bins)
    hist_diff = tf.math.square(hist_true - hist_pred)
    print(hist_diff)
    return tf.reduce_mean(hist_diff, axis=-1)

df = pd.read_csv("/data/pmurphy/spike_positions.csv", index_col=0)
# something funny with some drift_rate entries so drop/convert non negative ones
# df = df[(-1*df['drift_rate'].abs()) == df['drift_rate']].dropna()
df['drift_rate'] =-1* df['drift_rate'].abs()
df = df.drop([534])
# everything in MHz
df['spectral_extent'] /= 1e3
df['drift_rate'] /= 1e3
df['frequency_at_maximum_intensity'] /= 1e3
df['instantaneous_spectral_extent'] /= 1e3
df['minimum_frequency'] /= 1e6
df['maximum_frequency'] /= 1e6

data_path = Path("/data/pmurphy/spike_dataset/")
train_generator = DataGenerator(data_path/"train/", train=True, blank=False)
test_generator =  DataGenerator(data_path/"val/", train=False, blank=False)
blank_generator = DataGenerator(data_path/"blank/", train=True, blank=True)

burst_param_keys = df.loc[0].keys()
burst_param_keys = burst_param_keys.reindex(
                                        [*burst_param_keys.drop(
                                            [
                                            'time_at_maximum_intensity',
                                                'minimum_frequency',
                                                'maximum_frequency']).to_list(), 
                                         'time_at_maximum_intensity'])[0]

train_ds = tf.data.Dataset.from_generator(train_generator,
                                          output_signature=(
                                              tf.TensorSpec(shape=(64,64,4),
                                                            dtype=tf.float64),
                                              (tf.TensorSpec(shape=(64,64),
                                                            dtype=tf.float64),
                                              tf.TensorSpec(shape=(len(burst_param_keys)), dtype=tf.float64),
                                              tf.TensorSpec(shape=(), dtype=tf.float64)
                                              )
                                          )
                                         )
test_ds = tf.data.Dataset.from_generator(test_generator,
                                          output_signature=(
                                              tf.TensorSpec(shape=(64,64,4),
                                                            dtype=tf.float64),
                                              (tf.TensorSpec(shape=(64,64),
                                                            dtype=tf.float64),
                                              tf.TensorSpec(shape=(len(burst_param_keys)), dtype=tf.float64),
                                              tf.TensorSpec(shape=(), dtype=tf.float64)
                                               )
                                          )
                                         )

blank_ds = tf.data.Dataset.from_generator(blank_generator,
                                          output_signature=(
                                              tf.TensorSpec(shape=(64,64,4),
                                                            dtype=tf.float64),
                                              (tf.TensorSpec(shape=(64,64),
                                                            dtype=tf.float64),
                                              tf.TensorSpec(shape=(len(burst_param_keys)), dtype=tf.float64),
                                              tf.TensorSpec(shape=(), dtype=tf.float64)
                                               )
                                          )
                                         )

train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(train_generator.__len__()))
test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(test_generator.__len__()))
blank_ds = blank_ds.apply(tf.data.experimental.assert_cardinality(blank_generator.__len__()))
train_ds = train_ds.concatenate(blank_ds)

train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds, shuffle=True)

load = np.load(Path(data_path/"train/data_spike_id_000_augment_000.npy"), allow_pickle=True).item()
time_array = load['time']
freq_array = load['frequency']
pix_df = freq_array[1] - freq_array[0]
pix_dt = (Time(time_array[1]) - Time(time_array[0])).sec

# recreate UNET from segmentation models
ResNet34,_ = Classifiers.get('resnet34')
resnet34 = ResNet34(input_shape=(64, 64, 3), weights='imagenet', include_top=False, name="resnet_encoder")
classes = 1
activation = 'sigmoid'
n_neurons = 30
for layer in resnet34.layers:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

resnet34_input_ = resnet34.input

skip_layer_names = ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')
skips = ([resnet34.get_layer(name=i).output for i in skip_layer_names])
down_stack = tf.keras.Model(inputs=resnet34.input, outputs=skips, name="down_stack")

inputs = tf.keras.layers.Input(shape=[64, 64,  4], name='data_in')
clean_inputs = drop_channel(3)(inputs)
skips = down_stack(clean_inputs)


encoded = resnet34(clean_inputs)
encoded = tf.keras.layers.Flatten()(encoded)
r = tf.keras.layers.BatchNormalization()(encoded)
n_hidden = 5
for h in range(n_hidden):
    r =  tf.keras.layers.Dense(n_neurons, use_bias=False, kernel_initializer="he_normal", name="hidden_{}".format(h))(r)
    r = tf.keras.layers.BatchNormalization()(r)
    r = tf.keras.layers.LeakyReLU()(r)

objectness_output = tf.keras.layers.Dense(1)(r)
objectness_output = tf.keras.activations.sigmoid(objectness_output)
regressor_output = tf.keras.layers.Dense(len(burst_param_keys))(r)

# Create the feature extraction model
x = resnet34(clean_inputs)
n_upsample_blocks=5
decoder_filters=(256, 128, 64, 32, 16)
decoder_block = DecoderUpsamplingX2Block
for i in range(n_upsample_blocks):

    if i < len(skips):
        skip = skips[i]
    else:
        skip = None

    x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=True)(x, skip)


x = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
x = tf.keras.layers.Activation(activation, name=activation)(x)

regressor_model = tf.keras.models.Model(inputs, (x, regressor_output, objectness_output), name='UNET_regressor')

batch_size = 32
reguliser = 'EarlyStopping'

model_config = 'batch_size_' + str(batch_size) +\
        '_regularisation_' + reguliser +\
        '_hidden_layers_' + str(n_hidden) +\
        '_neurons_per_layer_' + str(n_neurons) +\
        '_n_params_' + str(len(burst_param_keys)) +\
        '_activation_lrelu_objectness_moreblank_loss_weights'

epochs = 100

lrate0 = 1e-2
decay_steps = (epochs/10) * len(train_ds)
alpha = 1e-2
warmup_target = 1e-1
# decay_rate = 0.5
warmup_steps = 5*len(train_ds)

lrate = CosineDecay(lrate0, decay_steps, alpha, warmup_target=warmup_target, warmup_steps=warmup_steps)
# linear_lrate = LinearLRateIncrease(1e-5, 10, epochs*len(train_ds))
optimizer= tf.keras.optimizers.Adam(learning_rate=lrate)
loss = [tf.keras.losses.BinaryCrossentropy(name="segmentation_loss"),
        tf.keras.losses.MeanSquaredError(name="regression_loss"),
        tf.keras.losses.BinaryCrossentropy(name="objectness_loss"),
        histogram_loss]
loss_weights = [0.75, 1, 0.9,1]
metrics = [[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)], [tf.keras.metrics.RootMeanSquaredError()],[]]

callback_start_epoch = (warmup_steps/len(train_ds))+(decay_steps/len(train_ds))
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, start_from_epoch=callback_start_epoch)

regressor_model.compile(optimizer=optimizer,
                        loss=loss,
                        loss_weights=loss_weights,
                        metrics=metrics)

regressor_history = regressor_model.fit(train_ds,
                epochs=epochs,
                validation_data=(test_ds),
                callbacks=[callback]
                )

fig, ax = plt.subplots(4,1, figsize=(9,9), sharex=True)
ax[0].plot(regressor_history.epoch, regressor_history.history['loss'])
ax[0].plot(regressor_history.epoch, regressor_history.history['val_loss'])
ax[1].plot(regressor_history.epoch, regressor_history.history['sigmoid_loss'])
ax[1].plot(regressor_history.epoch, regressor_history.history['val_sigmoid_loss'])
ax[2].plot(regressor_history.epoch, regressor_history.history['dense_1_loss'])
ax[2].plot(regressor_history.epoch, regressor_history.history['val_dense_1_loss'])
ax[3].plot(regressor_history.epoch, regressor_history.history['tf.math.sigmoid_loss'])
ax[3].plot(regressor_history.epoch, regressor_history.history['val_tf.math.sigmoid_loss'])
ax[2].set_xlabel("Epoch")
ax[0].set_ylabel("Total Loss")
ax[1].set_ylabel("Segmentation Loss")
ax[2].set_ylabel("Regression Loss")
ax[3].set_ylabel("Objectness Loss")
# ax[2].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')
fig.suptitle(model_config)
plt.savefig(data_path/"pngs/loss_vs_epoch_{}.png".format(model_config))

fig, ax = plt.subplots(2,1, figsize=(9,7), sharex=True)
ax[0].plot(regressor_history.epoch, regressor_history.history['sigmoid_binary_io_u'])
ax[0].plot(regressor_history.epoch, regressor_history.history['val_sigmoid_binary_io_u'])
ax[1].plot(regressor_history.epoch, regressor_history.history['dense_1_root_mean_squared_error'])
ax[1].plot(regressor_history.epoch, regressor_history.history['val_dense_1_root_mean_squared_error'])
ax[1].set_xlabel("Epoch")
ax[0].set_ylabel("Segmentation IOU")
ax[1].set_ylabel("Regression RMS")
# ax[1].set_xscale('log')

# ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.suptitle(model_config)
plt.savefig(data_path/"pngs/metric_vs_epoch_{}.png".format(model_config))

