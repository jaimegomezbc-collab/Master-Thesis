#from single_synth import generate_cet1
import os
import numpy as np
import pandas as pd
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
        #print(f"Image at {image_path} is too small or empty. Skipping.")
        return None

    # Decide whether to keep the ground truth based or the synthetic image
    if True:
        # Keep the ground truth based image
        image = np.rot90(image, k=-1)
        Image.fromarray(image).save(os.path.join(output_dir, os.path.basename(image_path)))
    else:
        print(f"Generating synthetic image for {image_path}")
        # generate_cet1(os.path.basename(image_path), os.path.dirname(os.path.dirname(image_path)), output_dir)
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
    labels = pd.read_csv("C:\\Users\\jaime\\OneDrive\\Downloads\\archive (1)\\MU-Diff images\\meta_data.csv")
    input_path = "C:\\Users\\jaime\\OneDrive\\Downloads\\archive (1)\\MU-Diff images"  # Replace with your input directory path
    output_path = "C:\\Users\\jaime\\OneDrive\\Downloads\\archive (1)\\MU-Diff images\\processed"  # Replace with your desired output directory path
    for subfolder in os.listdir(input_path):
        subfolder_path = os.path.join(input_path, subfolder)
        suboutput_path = os.path.join(output_path, subfolder)
        os.makedirs(suboutput_path, exist_ok=True)
        cet1_path = os.path.join(subfolder_path,'t1ce')
        if os.path.isdir(cet1_path):
            print(f"Folder: {cet1_path}")
            for file_name in os.listdir(cet1_path):
                label = labels[labels['slice_path'] == file_name[:-4]]['target'].values
                if label.size == 0:
                    print(f"Label for {file_name} not found. Skipping.")
                    continue
                if label == 1:
                    expected_output_file = os.path.join(suboutput_path, 'tumour', file_name)    
                    file_output_path = os.path.join(suboutput_path, 'tumour')
                    os.makedirs(file_output_path, exist_ok=True)
                else:
                    expected_output_file = os.path.join(suboutput_path, 'healthy', file_name)
                    file_output_path = os.path.join(suboutput_path, 'healthy')
                    os.makedirs(file_output_path, exist_ok=True)
                if os.path.exists(expected_output_file):
                    continue
                file_path = os.path.join(cet1_path, file_name)
                create_classifier_dataset(file_path, file_output_path)