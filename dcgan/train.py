import tensorflow as tf
from dcgan.config.constants import LATENT_DIM, BATCH_SIZE
from dcgan.model import cross_entropy, generator_optimizer, discriminator_optimizer
from dcgan.utils import generate_and_save_images
import os
import matplotlib.pyplot as plt

@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training = True)

        real_output = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training = True)

        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs, generator, discriminator):
    seed = tf.random.normal([16, LATENT_DIM])
    os.makedirs('generated_images', exist_ok = True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch, generator, discriminator)

        print(f"Generator loss: {g_loss.numpy():.4f} | Discriminator loss: {d_loss.numpy():.4f}")
        generate_and_save_images(generator, epoch, seed)

def generate_images(n, generator):
    noise = tf.random.normal([n, LATENT_DIM])
    images = generator(noise, training = False)
    images = (images + 1)/2.0

    for i in range(n):
        plt.imshow(images[i])
        plt.axis('off')
        plt.savefig(f"outputs/samples/sample_{i}.png")
        plt.close()