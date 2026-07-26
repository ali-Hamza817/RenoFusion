import subprocess
print("Running TotalSegmentator test...")
cmd = ["TotalSegmentator", "-i", "datasets/dataset_3_nifti/TCGA-BP-4166_0000.nii.gz", "-o", "datasets/dataset_3_nifti/TCGA-BP-4166_mask.nii.gz", "-ml", "-rs", "kidney_left", "kidney_right", "--fast"]
subprocess.run(cmd)
