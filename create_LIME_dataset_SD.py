import os
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
def save_nifti_middle_slices_as_npy(input_folder, output_folder, dtype=np.float32):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    nii_folds = os.listdir(input_folder)

    if len(nii_folds) == 0:
        print(f"No .nii.gz files found in {input_folder}")
        return

    for nii_path in nii_folds:
        patient_name = nii_path[11:18]
        t1 = nib.load(str(os.path.join(os.path.join(input_folder,nii_path), f"UCSD-PTGBM-{patient_name}_T1pre.nii.gz")))
        t2 = nib.load(
            str(os.path.join(os.path.join(input_folder, nii_path), f"UCSD-PTGBM-{patient_name}_T2.nii.gz")))
        flair = nib.load(
            str(os.path.join(os.path.join(input_folder, nii_path), f"UCSD-PTGBM-{patient_name}_FLAIR.nii.gz")))
        cet1 = nib.load(
            str(os.path.join(os.path.join(input_folder, nii_path), f"UCSD-PTGBM-{patient_name}_T1post.nii.gz")))
        mask = nib.load(str(os.path.join(os.path.join(input_folder,nii_path), f"UCSD-PTGBM-{patient_name}_BraTS_tumor_seg.nii.gz")))

        mask_data = mask.get_fdata().astype(dtype)
        tumour_slices =[]

        if mask_data.ndim != 3:
            print(f"Skipping {nii_path.name}: expected 3D volume, got shape {data.shape}")
            continue

        num_slices = mask_data.shape[0]

        for slice_idx in range(num_slices):
            mask_2d = mask_data[slice_idx,:, :]

            if np.max(mask_2d) == 3:
                tumour_slices.append(slice_idx)
        if len(tumour_slices)==0:
            print(f"Skipping image {nii_path}. No enhancement regions found")
            continue
        elif len(tumour_slices)==1:
            middle_slice = tumour_slices[0]
        else:
            middle_slice = tumour_slices[len(tumour_slices)//2]
        mask_2d = mask_data[middle_slice,:,:]
        mask_2d_contrast = mask_2d==3
        np.save(os.path.join(os.path.join(output_folder,"enhanced_regions_mask"), f"{nii_path}_slice{middle_slice}.npy"), np.rot90(mask_2d_contrast))
        t1_data = t1.get_fdata().astype(dtype)
        np.save(os.path.join(os.path.join(output_folder,"t1"), f"{nii_path}_slice{middle_slice}.npy"), np.rot90(t1_data[middle_slice]))
        t2_data = t2.get_fdata().astype(dtype)
        np.save(os.path.join(os.path.join(output_folder,"t2"), f"{nii_path}_slice{middle_slice}.npy"), np.rot90(t2_data[middle_slice]))
        flair_data = flair.get_fdata().astype(dtype)
        np.save(os.path.join(os.path.join(output_folder,"flair"), f"{nii_path}_slice{middle_slice}.npy"), np.rot90(flair_data[middle_slice]))
        cet1_data = cet1.get_fdata().astype(dtype)
        np.save(os.path.join(os.path.join(output_folder,"cet1"), f"{nii_path}_slice{middle_slice}.npy"), np.rot90(cet1_data[middle_slice]))



if __name__ == "__main__":
    input_folder = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/train"
    output_folder = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/middle_slices"

    save_nifti_middle_slices_as_npy(input_folder, output_folder)