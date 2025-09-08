from dcgan.utils import load_dataset
from dcgan.model import build_discriminator, build_generator
from dcgan.train import generate_images, train
from dcgan.config.constants import EPOCHS
import os

def main():
    dataset = load_dataset('data')

    generator = build_generator()
    discriminator = build_discriminator()

    train(dataset, EPOCHS, generator, discriminator)

    generate_images(n = 5, generator = generator)

    generator.save("outputs/models/generator.keras")
    discriminator.save("outputs/models/discriminator.keras")

    

if __name__ == "__main__":
    main()