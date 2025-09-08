from utils import *
from model import *
from train import *
from config.constants import *
import os

dataset = load_dataset('data/archive/images')

generator = build_generator()
discriminator = build_discriminator()

train(dataset, EPOCHS)

os.makedirs('../outputs/models/generator', exist_ok=True)
os.makedirs('../outputs/models/generator', exist_ok=True)

generator.save("../outputs/models/generator")
discriminator.save("../outputs/models/discriminator")

generate_images(n=5)