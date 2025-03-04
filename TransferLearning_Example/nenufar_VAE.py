#!/usr/bin/env python

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from segmentation_models import Unet
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

from data_gen_test import ltime, lfreq, create_train_test_val_ds, drop_channel

# from Tensorflow documentation
# https://www.tensorflow.org/tutorials/generative/cvae

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, n_channels):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.n_channels = n_channels
    # self.unet = unet
    # bottle_neck_layer = unet.layers[:[layer.name for layer in unet.layers].index('relu1')]
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(256,
                                                    256,
                                                    self.n_channels)),
            drop_channel(3),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=512, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )
  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstruction = self.decoder(z)
    return reconstruction 
   
  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
  

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  x = x[:,:,:,np.newaxis]
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.cast(x_logit, np.float32), labels=tf.cast(x, np.float32))
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_sample, latent_dim):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(tf.transpose(predictions[i, :, :, 0]), cmap='viridis')
    # plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}_{}_latentdims.png'.format(epoch, latent_dim))
  # plt.show()


# Alternative implementation
# From https://keras.io/examples/generative/vae

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        # self.build((None, 256, 256, n_channels))
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print(data)
            print(reconstruction)
            stokesI = data[:,:,:,0][:,:,:,np.newaxis]
            print(stokesI)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(stokesI, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
      z_mean, z_log_var, z = self.encoder(inputs)
      reconstruction = self.decoder(z)
      stokesI = inputs[:,:,:,0][:,:,:,np.newaxis]
      reconstruction_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.keras.losses.binary_crossentropy(stokesI, reconstruction), axis=(1, 2)
          )
      )
      kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
      kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
      total_loss = reconstruction_loss + kl_loss
      self.add_loss(total_loss)
      self.add_loss(reconstruction_loss)
      self.add_loss(kl_loss)
      return reconstruction


def alt_generate_and_save_images(model, epoch, test_sample, latent_dim, gt=False):
  if gt:
    fig = plt.figure(figsize=(9, 9))

    for i in range(test_sample.shape[0]):
      plt.subplot(4, 4, i + 1)
      plt.imshow(tf.transpose(test_sample[i,  :, :, 0]), cmap='viridis')
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('VGG_bn_image_{}_latentdims_alt_gt.png'.format(latent_dim))  
    
  else:
    z_mean, z_logvar, z = model.encoder(test_sample)
    
    predictions = model.decoder(z)
    fig = plt.figure(figsize=(9, 9))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i + 1)
      plt.imshow(tf.transpose(predictions[i, :, :, 0]), cmap='viridis')
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('VGG_bn_image_at_epoch_{:04d}_{}_latentdims_alt.png'.format(epoch, latent_dim))



class plot_on_epoch(tf.keras.callbacks.Callback):
  def __init__(self, plot_epochs=10):
    super().__init__()
    self.plot_epochs = plot_epochs

  def on_epoch_end(self, epoch, logs=None):
     if epoch % self.plot_epochs == 0: 
        alt_generate_and_save_images(self.model, epoch, test_sample, latent_dim)

# some other tensorflow tutorial thing
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
# https://www.tensorflow.org/tutorials/images/segmentation#define_the_model
def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def vae_unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


class choose_channel(layers.Layer):
    # remove channel before it goes into the network
    def __init__(self, last_chan):
        super(choose_channel, self).__init__()
        self.chan = chan
        self.build((None, 256, 256, 4))
    def __call__(self, inputs):
        return inputs[:,:,:,self.chan]


tf.keras.backend.clear_session()
num_classes=2

backbone = "resnet34"
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
window_size = 256
n_channels = 4

# unet = Unet(backbone_name=backbone,
#                         # input_shape=(None, None, n_channels),
#                         classes=num_classes,
#                         encoder_weights='imagenet',
#                         encoder_freeze=True,)
# # unet.summary()

# inp = layers.Input(shape=(None, None, n_channels))
# enc = drop_channel(3)(inp) # drop last channel

# for layer in unet.layers[:[layer.name for layer in unet.layers].index('relu1')]:
#   enc = layer(enc)

# encoder = tf.keras.Model(inp, enc)
# encoder.summary()
# out = unet(ld)


# bottle_neck_layer = unet.layers[[layer.name for layer in unet.layers].index('relu1')]
# full_input = tf.keras.Model(inp, out)
# bottle_neck_output = tf.keras.Model(unet.input, bottle_neck_layer.output)


# # print(encoder)
# print(bottle_neck_layer, type(bottle_neck_layer))

# out = unet(ld)

# model = models.Model(inp, out, name=unet.name)

test_size = 0.2
k = 0
n_splits = 5
sss = StratifiedShuffleSplit(n_splits=n_splits,
                             test_size=test_size,
                             random_state=42)

train_ds, test_ds = create_train_test_val_ds(ltime[:],
                                            lfreq[:],
                                            sss,
                                            window_size,
                                            k,
                                            batch_size=batch_size,
                                            backbone='resnet34',
                                            n_channels=n_channels)

latent_dim = 10000
print(train_ds)

encoder_inputs = tf.keras.Input(shape=(256, 256, n_channels), name="encoder_inputs")

base_model = ResNet50(input_tensor=encoder_inputs, include_top=False, weights=None)
# base_model = VGG16(input_tensor=encoder_inputs, include_top=False, weights=None)
# for layer in base_model.layers:
#   print(layer.name)

# base_model.summary()

# layer_names = [
#   "conv1_relu",
#   "conv2_block3_out",
#   "conv3_block4_out",
#   "conv4_block6_out",
#   "conv5_block3_out",
# ]

# layer_names = [
#   "block1_pool",
#   "block2_pool",
#   "block3_pool",
#   "block4_pool",
#   "block5_pool",
# ]

# base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
# down_stack = tf.keras.models.Seqeuntial()
# down_stack.add(tf.keras.Model(inputs=encoder_inputs, outputs=base_model_outputs))
# down_stack.add(layers.Flatten())
# down_stack.add(layers.Dense(1000, activation='relu'))

# down_stack.trainable = False


# down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name="down_stack")

# up_stack = [
#     upsample(1024, 3),
#     upsample(512, 3),
#     upsample(256, 3),
#     upsample(128, 3),
#     upsample(64, 3), 
# ]


x = drop_channel(3)(encoder_inputs)
x = drop_channel(2)(x)
x = drop_channel(1)(x) # 256 x 256 x 1


# skips = down_stack(base_model.input)
# # x = skips[-1]
# x = base_model(base_model.input)
# print(skips)
# skips = reversed(skips[:-1])
# x = base_model(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(32, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(1024, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(3, 2)(x) # 128 x 128 x 64

x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(3, 2)(x) # 64 x 64 x 128

x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(3, 2)(x) # 32 x 32 x 256

x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.MaxPool2D(3, 2)(x) # 16 x 16 x 512

x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(3, 2)(x) # 8 x 8 x 512

x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
# encoder = tf.keras.Model(base_model.input, [z_mean, z_log_var, z], name="encoder")
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = tf.keras.Input(shape=(latent_dim,), name="latent_inputs")

x = layers.Dense(8 * 8 * 512, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 512))(x)

x = layers.Conv2DTranspose(512, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(512, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)


# x = layers.Conv2DTranspose(1024, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(512, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)

# x = layers.Dense(4 * 4 * 1024, activation="relu")(latent_inputs)
# x = layers.Reshape((4, 4, 1024))(x)
# x = layers.Conv2DTranspose(1024, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(512, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.BatchNormalization()(x)


# for up, skip in zip(up_stack, skips):
#   x = up(x)
#   print(x)
#   concat = tf.keras.layers.Concatenate()
#   x = concat([x, skip])
#   print(x)
# for up in up_stack:
#   x = up(x)






decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
print(latent_inputs, decoder_outputs)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

vae = VAE(encoder, decoder)
vae.compute_output_shape(input_shape=(None, 256, 256, n_channels))
vae.summary()
vae.compile(optimizer=tf.keras.optimizers.Adam())
epochs = 1001
model_name = "/data/pmurphy/VAE_{}_latentdim_{}_epoch_VGG_bn".format(latent_dim, epochs)
print(model_name)
# train_spectra = []
# for train_x, _ in train_ds:
#     train_spectra.append(train_x[:,:,:,0])
# train_spectra = list(map(lambda x: x[0][:,:,:,0], train_ds))
# train_spectra = np.concatenate(train_spectra, axis=0)
# train_spectra = train_spectra[:,:,:,np.newaxis]



# optimizer = tf.keras.optimizers.Adam(1e-4)

# epochs = 100
# # set the dimensionality of the latent space to a plane for visualization later
# latent_dim = 5
num_examples_to_generate = 16

# # keeping the random vector constant for generation (prediction) so
# # it will be easier to see the improvement.
# random_vector_for_generation = tf.random.normal(
#     shape=[num_examples_to_generate, latent_dim])
# model = CVAE(latent_dim, 1)


# # Pick a sample of the test set for generating output images
# # assert batch_size >= num_examples_to_generate
for test_batch in test_ds.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=model_name+'_tboard'), plot_on_epoch(epochs//10)]
alt_generate_and_save_images(vae, epochs, test_sample, latent_dim, gt=True)
vae.fit(train_ds, epochs=epochs, batch_size=32, verbose=2, callbacks=callbacks)


# generate_and_save_images(model, 0, test_sample, latent_dim)

# for epoch in range(1, epochs + 1):
#   start_time = time.time()
#   for train_x, _ in train_ds:
#     train_step(model, train_x[:,:,:,0], optimizer)
#   end_time = time.time()

#   loss = tf.keras.metrics.Mean()
#   for test_x, _ in test_ds:
#     loss(compute_loss(model, test_x[:,:,:,0]))
#   elbo = -loss.result()

#   print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
#         .format(epoch, elbo, end_time - start_time))
#   if epoch % 10 == 0:
#     generate_and_save_images(model, epoch, test_sample, latent_dim)

vae.save(model_name)
