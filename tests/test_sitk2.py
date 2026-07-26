import SimpleITK as sitk
import nibabel as nib
import numpy as np

img_path = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698/ct_scan.nii.gz"
mask_path = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698/tumor_mask.nii.gz"
out_path = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698/fixed_mask.nii.gz"

try:
    print("Loading image with sitk...")
    img = sitk.ReadImage(img_path)
    print("Loading mask with nibabel...")
    mask_nib = nib.load(mask_path)
    
    # Simple fix: get affine from the image (which sits perfectly) and apply to mask
    img_nib = nib.load(img_path)
    fixed_mask = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)
    nib.save(fixed_mask, out_path)
    
    print("Loading fixed mask with sitk...")
    sitk_mask = sitk.ReadImage(out_path)
    print("Success! Fixed mask loaded in sitk.")
except Exception as e:
    print("Error:", e)
