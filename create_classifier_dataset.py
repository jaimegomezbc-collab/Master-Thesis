from single_synth import generate_cet1
import os
import numpy as np
import torch
from PIL import Image

def create_classifier_dataset(image_path,output_dir):
    """
    Create a dataset for training a classifier from the given image path and output path.

    Args:
        image_path (str): Path to the input images.
        output_dir (str): Path where the processed dataset will be saved.
    """
    # Load the image from the specified path
    image = load_image(image_path)
    if image is None:
        print(f"Image at {image_path} is too small or empty. Skipping.")
        return None

    # Decide whether to keep the ground truth based or the synthetic image
    if np.random.rand() < 0.5:
        # Keep the ground truth based image
        image = np.rot90(image, k=-1)
        Image.fromarray(image).save(os.path.join(output_dir, os.path.basename(image_path)))
    else:
        print(f"Generating synthetic image for {image_path}")
        generate_cet1(os.path.basename(image_path), os.path.dirname(os.path.dirname(image_path)), output_dir)
    return None

def load_image(image_path, size=(256, 256)):
    """
    Loads an image from the specified path, preprocesses it, and returns it as a tensor with shape (1, 1, 256, 256).
    """
    
    img = Image.open(image_path).convert("L")  # Convert to grayscale ('L' mode for single channel)
    img = img.resize(size, Image.BILINEAR)
    img_np = np.array(img)  # Convert to numpy array for min-max scaling
    if np.sum(img_np>0) < 1000:
        return None
    else:
        return img

if __name__ == "__main__":
    # Example usage
    image_paths = [r"C:/Users/jaime/OneDrive/Downloads/archive (1)/MU-Diff images/train/t1ce/volume_100_slice_0.jpg", r"C:/Users/jaime/OneDrive/Downloads/archive (1)/MU-Diff images/train/t1ce/volume_100_slice_92.jpg", r"C:/Users/jaime/OneDrive/Downloads/archive (1)/MU-Diff images/train/t1ce/volume_95_slice_89.jpg"]  # Replace with your image paths
    output_dir = r"C:/Users/jaime/OneDrive/Downloads/archive (1)/Code_test"  # Replace with your desired output directory
    os.makedirs(output_dir, exist_ok=True)
    for image_path in image_paths:
        create_classifier_dataset(image_path, output_dir)