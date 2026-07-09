from single_synth import generate_cet1
import os
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing as mp

def build_jobs(input_path, output_path, labels):
    jobs = []

    for subfolder in os.listdir(input_path):
        subfolder_path = os.path.join(input_path, subfolder)
        suboutput_path = os.path.join(output_path, subfolder)
        os.makedirs(suboutput_path, exist_ok=True)

        cet1_path = os.path.join(subfolder_path, "t1ce")
        if not os.path.isdir(cet1_path):
            continue

        print(f"Folder: {cet1_path}")

        for file_name in os.listdir(cet1_path):
            label = labels.loc[labels["slice_path"] == file_name[:-4], "target"].values
            if label.size == 0:
                print(f"Label for {file_name} not found. Skipping.")
                continue

            if label[0] == 1:
                file_output_path = os.path.join(suboutput_path, "tumour")
            else:
                file_output_path = os.path.join(suboutput_path, "healthy")

            os.makedirs(file_output_path, exist_ok=True)
            expected_output_file = os.path.join(file_output_path, file_name)

            if os.path.exists(expected_output_file):
                continue

            file_path = os.path.join(cet1_path, file_name)
            jobs.append((file_path, file_output_path))

    return jobs


def worker(job_queue):
    while True:
        try:
            file_path, file_output_path = job_queue.get_nowait()
        except Exception:
            break

        try:
            # Remove the lock if you want true concurrent GPU access.
            # Keep it if you want 2 workers for CPU-side prep but only 1 GPU inference at once.
            with gpu_lock:
                create_classifier_dataset(file_path, file_output_path)

            print(f"Done: {file_path}")
        except Exception as e:
            print(f"Failed: {file_path} -> {e}")


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
    if random.random()<0.5:
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
    labels = pd.read_csv("/cs/student/project_msc/2025/aibh/jgomezbe/meta_data.csv")
    input_path = "/cs/student/project_msc/2025/aibh/jgomezbe/images"  # Replace with your input directory path
    output_path = "/cs/student/project_msc/2025/aibh/jgomezbe/classifier_images"  # Replace with your desired output directory path
    jobs = build_jobs(input_path, output_path, labels)
    print(f"Total jobs: {len(jobs)}")

    manager = mp.Manager()
    job_queue = manager.Queue()

    for job in jobs:
        job_queue.put(job)

    num_workers = 2
    processes = []

    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(job_queue,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()