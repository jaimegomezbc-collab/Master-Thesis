import os
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
def save_nifti_slices_as_npy(input_folder, output_folder, dtype=np.float32):
    input_folder = Path(input_folder)
    output_healthy = os.path.join(output_folder,'healthy')
    output_tumour = os.path.join(output_folder, 'tumour')
    output_healthy = Path(output_healthy)
    output_tumour = Path(output_tumour)
    output_healthy.mkdir(parents=True, exist_ok=True)
    output_tumour.mkdir(parents=True, exist_ok=True)

    nii_folds = os.listdir(input_folder)

    if len(nii_folds) == 0:
        print(f"No .nii.gz files found in {input_folder}")
        return

    for nii_path in nii_folds:
        patient_name = nii_path[11:18]
        img = nib.load(str(os.path.join(os.path.join(input_folder,nii_path), f"UCSD-PTGBM-{patient_name}_T1post.nii.gz")))
        mask = nib.load(str(os.path.join(os.path.join(input_folder,nii_path), f"UCSD-PTGBM-{patient_name}_BraTS_tumor_seg.nii.gz")))

        mask_data = mask.get_fdata().astype(dtype)
        data = img.get_fdata().astype(dtype)

        if data.ndim != 3:
            print(f"Skipping {nii_path.name}: expected 3D volume, got shape {data.shape}")
            continue

        num_slices = data.shape[0]

        for slice_idx in range(num_slices):
            slice_2d = data[slice_idx,:, :]
            mask_2d = mask_data[slice_idx,:, :]
            out_name = f"{patient_name}_{slice_idx:03d}.npy"

            if np.max(mask_2d) >= 2:
                out_path = output_tumour
            else:
                out_path = output_healthy

            slice_2d = np.rot90(slice_2d)
            np.save(os.path.join(out_path, out_name), slice_2d)

        print(f"Saved {num_slices} slices from {nii_path}")


if __name__ == "__main__":
    input_folder = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/train"
    output_folder = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/MONAI_data/train"

    save_nifti_slices_as_npy(input_folder, output_folder)