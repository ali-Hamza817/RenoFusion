import SimpleITK as sitk
import sys
mask_path = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698/tumor_mask.nii.gz"
try:
    img = sitk.ReadImage(mask_path)
    print("Success")
except Exception as e:
    print("Error:", e)
