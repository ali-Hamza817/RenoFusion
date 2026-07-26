import os
import tempfile
import shutil
import subprocess

# Let's copy a validation case to a tmp dir to simulate the upload
val_dir = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698"
img_path = os.path.join(val_dir, "ct_scan.nii.gz")

tmp_dir = tempfile.mkdtemp()
tmp_img = os.path.join(tmp_dir, "image.nii.gz")
shutil.copy(img_path, tmp_img)

cmd = ["TotalSegmentator", "-i", tmp_img, "-o", tmp_dir, "--fast", "-ta", "kidney_vessels"]
print("Running TotalSegmentator...")
subprocess.run(cmd, check=True)

mask_path_l = os.path.join(tmp_dir, 'kidney_left.nii.gz')
mask_path_r = os.path.join(tmp_dir, 'kidney_right.nii.gz')

import nibabel as nib
img_nib = nib.load(tmp_img)
for m_path in [mask_path_l, mask_path_r]:
    if os.path.exists(m_path):
        mask_nib = nib.load(m_path)
        fixed_mask = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)
        nib.save(fixed_mask, m_path)

import SimpleITK as sitk
def get_mask_volume(p):
    if not os.path.exists(p): 
        print(f"{p} does not exist.")
        return 0
    try:
        vol = sitk.GetArrayFromImage(sitk.ReadImage(p)).sum()
        print(f"Volume of {p}: {vol}")
        return vol
    except Exception as e:
        print(f"Error reading {p}: {e}")
        return 0

vol_l = get_mask_volume(mask_path_l)
vol_r = get_mask_volume(mask_path_r)
print(f"Total volume: left {vol_l}, right {vol_r}")
