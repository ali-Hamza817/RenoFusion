import nibabel as nib
img_path = '/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4712/ct_scan.nii.gz'
img_nib = nib.load(img_path)
print("Img shape:", img_nib.shape)
