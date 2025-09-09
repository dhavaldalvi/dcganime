from dcgan.utils import load_dataset, make_video
from dcgan.model import build_discriminator, build_generator
from dcgan.train import generate_images, train
from dcgan.config.constants import EPOCHS

def main():
    # Loading dataset
    dataset = load_dataset('data')
    
    # Building generator and discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Train generator and discriminator
    train(dataset, EPOCHS, generator, discriminator)
    
    # Generating sample images with trained generator
    generate_images(n = 5, generator = generator)
    
    # Saving model
    generator.save("outputs/models/generator.keras")
    discriminator.save("outputs/models/discriminator.keras")
    
    # Making video from saved images to show the progess of the model
    make_video('outputs/generated_images', 'outputs/generated_video/generated_video.mp4', fps=EPOCHS)

if __name__ == "__main__":
    main()