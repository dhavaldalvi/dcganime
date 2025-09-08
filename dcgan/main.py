from utils import *
from model import *
from train import *

dataset = load_dataset('data/archive/images')

generator = build_generator()
discriminator = build_discriminator()