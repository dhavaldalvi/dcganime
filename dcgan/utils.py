import os
import glob
import tensorflow as tf
from dcgan.config.constants import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
import matplotlib.pyplot as plt

def get_image_files(image_dir):
    extensions = ['jpg', 'jpeg', 'png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, f'*.{ext}')))
        #files.extend(glob.glob(os.path.join(image_dir, f'**/*.{ext}'), recursive=True))
    return files

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = (img - 0.5)*2
    return img

def load_dataset(image_dir):
    image_paths = get_image_files(image_dir)
    # image_paths = [str(p) for p in image_paths]  # Ensure all are strings
    # image_paths_tensor = tf.constant(image_paths, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training = False)
    predictions = (predictions + 1)/2.0

    fig = plt.figure(figsize = (4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.savefig(f"outputs/generated_images/images_at_epoch_{epoch:04d}.png")
    plt.close()
