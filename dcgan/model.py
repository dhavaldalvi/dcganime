import tensorflow as tf
from tensorflow.keras import layers
from dcgan.config.constants import LATENT_DIM, IMG_HEIGHT, IMG_WIDTH, CHANNELS

def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(7*7*512, use_bias = False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((7, 7, 512)),

        layers.Conv2DTranspose(256, 5, strides = 2, padding = 'same', use_bias = False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(128, 5, strides = 2, padding = 'same', use_bias = False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(CHANNELS, 5, strides = 2, padding = 'same', use_bias = False, activation = 'tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.Conv2D(64, 4, strides = 2, padding = 'same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, 4, strides = 2, padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, 4, strides = 2, padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1 = 0.5)
