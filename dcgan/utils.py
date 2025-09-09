import os
import glob
import tensorflow as tf
from dcgan.config.constants import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
import matplotlib.pyplot as plt
import cv2

# Get list of images
def get_image_files(image_dir):
    extensions = ['jpg', 'jpeg', 'png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, f'*.{ext}')))
    return files

# Data ingestion and preprocessing

# Preprocessing images
def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = (img - 0.5)*2
    return img

# Loading dataset
def load_dataset(image_dir):
    image_paths = get_image_files(image_dir)
    # image_paths = [str(p) for p in image_paths]  # Ensure all are strings
    # image_paths_tensor = tf.constant(image_paths, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Generate images from model and saving
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

# Make video from images
def make_video(image_path, video_path, fps=30):
    image_folder = image_path
    output_video = video_path
    
    # Get image file list and sort it
    images =  [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg','.png'))]
    images.sort()

    if len(images) == 0:
        raise ValueError('No images found in the folder')
    
    print(f"Found {len(images)} images.")

    # Read first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        raise ValueError(f"Failed to read the first image: {first_image_path}")
    height, width, layers = frame.shape
    print(f"Frame size: {width}x{height}")

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add each image to the video
    frame_count = 0
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        video.write(frame)
        frame_count += 1

    # Release the video writer
    video.release()
    print(f'video saved as {output_video} with {frame_count} frames.')