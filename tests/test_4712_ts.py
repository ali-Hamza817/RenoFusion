import subprocess
import tempfile
import os
import nibabel as nib
import SimpleITK as sitk

val_dir = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4712"
img_path = os.path.join(val_dir, "ct_scan.nii.gz")
tmp_dir = tempfile.mkdtemp()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"
try:
    print(f"Running TotalSegmentator on 4712...")
    subprocess.run([
        "TotalSegmentator", "-i", img_path, "-o", tmp_dir, "-rs", "kidney_left", "kidney_right", "--fast"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    
    for side in ['left', 'right']:
        p = os.path.join(tmp_dir, f"kidney_{side}.nii.gz")
        if os.path.exists(p):
            print(f"Fixing affine for {p}")
            img_nib = nib.load(img_path)
            mask_nib = nib.load(p)
            fixed_mask = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)
            nib.save(fixed_mask, p)
            
            img = sitk.ReadImage(p)
            vol = sitk.GetArrayFromImage(img).sum()
            print(f"Success! {side} Volume:", vol)

except subprocess.CalledProcessError as e:
    print("Error:", e.stderr.decode())
