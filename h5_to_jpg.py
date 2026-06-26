import h5py
import os
from PIL import Image

def open_h5_and_save_modalities(file_path, output_dir):
    """
    Open an h5 file and saves each modality in a separate jpg file.

    Parameters:
    file_path (str): The path to the h5 file.
    output_dir (str): The directory where the jpg files will be saved.

    """

    try:
        with h5py.File(file_path) as f:
            opened_h5 = f['image'][:]
        modalities = ['flair', 't1', 't1ce', 't2']
        for i, modality in enumerate(modalities):
            save_modalities_as_jpg(opened_h5, i, os.path.join(output_dir, modality), os.path.basename(file_path[:-3]))
        return None
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return None
    
def save_modalities_as_jpg(opened_h5, modality, output_dir,file_name):
    """
    Save a modality as a jpg file.

    Parameters:
    opened_h5 (array): The opened h5 file as an array (H, W, 4).
    modality (str): The modality to be saved.
    output_dir (str): The directory where the jpg file will be saved.
    file_name (str): The name of the file to be saved.

    """

    arr = opened_h5[:,:,modality]
    if (arr.max() - arr.min())>0:
        arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    else: 
        arr[:,:] = 0
    img = Image.fromarray(arr.astype('uint8'))
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(output_dir, f"{file_name}.jpg")):
        img.save(os.path.join(output_dir, f"{file_name}.jpg"))
    return None
    


if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(r"C:\Users\jaime\OneDrive\Downloads\archive (1)\MU-Diff data", split)
        output_split_dir = os.path.join(r"C:\Users\jaime\OneDrive\Downloads\archive (1)\MU-Diff images", split)
        os.makedirs(output_split_dir, exist_ok=True)
        for file_name in os.listdir(split_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(split_dir, file_name)
                open_h5_and_save_modalities(file_path, output_split_dir)