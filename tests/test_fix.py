import nibabel as nib
import SimpleITK as sitk

img_path = '/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698/ct_scan.nii.gz'
p = '/tmp/ts_test/kidney_left.nii.gz'

img_nib = nib.load(img_path)
mask_nib = nib.load(p)
fixed_mask = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)
nib.save(fixed_mask, p)

try:
    img = sitk.ReadImage(p)
    vol = sitk.GetArrayFromImage(img).sum()
    print("Success! Volume:", vol)
except Exception as e:
    print("Error:", e)
