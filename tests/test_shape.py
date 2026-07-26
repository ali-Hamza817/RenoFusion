import nibabel as nib
img_path = '/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698/ct_scan.nii.gz'
p = '/tmp/ts_test/kidney_left.nii.gz'
img_nib = nib.load(img_path)
mask_nib = nib.load(p)
print("Img shape:", img_nib.shape)
print("Mask shape:", mask_nib.shape)
print("Mask volume:", mask_nib.get_fdata().sum())
